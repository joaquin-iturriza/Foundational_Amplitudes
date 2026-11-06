"""The shared memmap cache (data_mmap.py, format 3) and the per-event keyed augmentation (keyed_aug.py), on CPU with
synthetic pools. Run: .venv-cpu/bin/python tests/test_mmap_key.py (or pytest).

What it checks (the end-to-end equivalence with the in-memory path on real pools is tests/test_memmap_data.py
section 2, on a site):
  keyed_aug   every transform is a Lorentz transform; an event's transform depends only on (seed, pool, index), not
              on the other events drawn with it; seeds and pools give different transforms; the distribution matches
              lloca's rand_lorentz(std_eta=0.1) (moments of the boost and rotation parts).
  key         the base key is unchanged by the pools' path and mtime (a copy elsewhere) and has no seed in it; it
              changes when a pool's content changes and with MMAP_VERSION.
  batches     a batch gathered from the cache equals the in-memory path's computation (centre-of-mass momenta,
              keyed augmentation of the whole pool, / momentum scale) for two seeds from one cache, bitwise, and
              does not depend on how the rows are grouped into batches.
  openers     two processes opening one cache at once read the same batches.
  canary      a cache whose array was overwritten after the build is refused on open.
  prune       the least recently used depth is pruned first, a recently used one never.
  fingerprint the code writing the cache is pinned to MMAP_VERSION.
"""
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import keyed_aug  # noqa: E402
import data_mmap as dm  # noqa: E402

# format_fingerprint() of MMAP_VERSION 3; a change to the cache-writing code must bump MMAP_VERSION and record its
# new fingerprint here
FINGERPRINT = {3: "91c077ac042017a79d0a10b4d11cf8ef1f9f7b029bba3808c1cfa223d0fd1b1e"}
ETA = np.diag([1.0, -1.0, -1.0, -1.0])


def test_lorentz_and_keys():
    n = 4000
    uids = np.full(n, 12345, dtype=np.uint64)
    idx = np.arange(n)
    T = keyed_aug.event_trafos(42, uids, idx)
    dev = np.abs(np.einsum("nji,jk,nkl->nil", T, ETA, T) - ETA).max()
    assert dev < 1e-10, dev
    sub = np.array([3, 17, 999, 2500])
    assert np.array_equal(keyed_aug.event_trafos(42, uids[sub], idx[sub]), T[sub])
    assert np.array_equal(keyed_aug.event_trafos(42, uids[sub][::-1], idx[sub][::-1]), T[sub][::-1])
    assert not np.allclose(keyed_aug.event_trafos(43, uids, idx), T)
    assert not np.allclose(keyed_aug.event_trafos(42, uids + np.uint64(1), idx), T)
    # the distribution against lloca's rand_lorentz(std_eta=0.1) (what experiment drew before, D290)
    from lloca.utils.rand_transforms import rand_lorentz
    N = 200000
    Tk = keyed_aug.event_trafos(7, np.full(N, 9, dtype=np.uint64), np.arange(N))
    torch.manual_seed(0)
    Tl = rand_lorentz((N,), dtype=torch.float64).numpy()
    for f, what in ((lambda t: t[:, 0, 0], "gamma"), (lambda t: (t[:, 0, 1:] ** 2).sum(1), "boost^2"),
                    (lambda t: t[:, 1:, 1:].reshape(len(t), -1).mean(1), "rotation block")):
        a, b = f(Tk), f(Tl)
        se = np.sqrt(a.var() / N + b.var() / N)
        assert abs(a.mean() - b.mean()) < 5 * se + 1e-12, (what, a.mean(), b.mean(), se)
        assert abs(a.std() - b.std()) < 0.03 * max(b.std(), 1e-12) + 1e-12, (what, a.std(), b.std())


def _pools(d, rng, names=("pa", "pb"), roles=("train", "val", "test"), n=600, P=4):
    paths = {r: {} for r in roles}
    for r in roles:
        for k, name in enumerate(names):
            E = rng.uniform(50, 500, size=n)
            pin = np.zeros((n, 2, 4))
            pin[:, 0] = np.stack([E, 0 * E, 0 * E, E], 1)
            E2 = E * rng.uniform(0.5, 2.0, size=n)
            pin[:, 1] = np.stack([E2, 0 * E, 0 * E, -E2], 1)
            out = rng.normal(size=(n, P - 2, 4)) * 100
            out[..., 0] = np.sqrt((out[..., 1:] ** 2).sum(-1))
            mom = np.concatenate([pin, out], 1).reshape(n, 4 * P)
            pdg = np.tile(np.array([11, -11, 1 + k, -1 - k] + [21] * (P - 4)), (n, 1))
            amp = rng.lognormal(size=(n, 1))
            p = os.path.join(d, f"{r}_{name}.npy")
            np.save(p, np.concatenate([mom, pdg, amp], 1))
            paths[r][name] = p
    return paths


def _fake_exp():
    from omegaconf import OmegaConf
    from experiment import AmplitudeExperiment
    e = types.SimpleNamespace(cfg=OmegaConf.create({"data": {"particle_order": "none", "seed": 42}}))
    e._boost_com = types.MethodType(AmplitudeExperiment._boost_com, e)
    return e


def _build(root, paths, names=("pa", "pb"), roles=("train", "val", "test")):
    exp = _fake_exp()
    c = {"names": list(names), "roles": list(roles), "paths": paths}
    kdoc = dm.base_key_doc(exp.cfg, names, roles, paths)
    need = dm._need({r: None for r in roles}, names, roles, paths)
    d = os.path.join(root, dm._hash(kdoc), dm._depth_name({r: None for r in roles}, roles))
    meta = dm._locked_build(d, dm._base_bytes(need, names, roles, paths)[0], root,
                            lambda st: dm._build_base(exp, c, st, need, kdoc))
    return exp, d, meta, kdoc


def _source(d, meta, seed, rows, mom_div=37.5):
    U = np.unique(np.concatenate([[p for p, _ in s["pdg_first"]] for s in meta["segs"]])).astype(np.int64)
    lut = np.tile(np.arange(1, len(U) + 1), (2, 1))
    order_rows = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    return dm.MmapSource(d, d, meta, rows, seed, mom_div, False, U, lut, order_rows, None, 1e-6, 3)


def _reference(exp, paths, meta, seed, mom_div=37.5):
    """the in-memory path's momenta: per pool, centre-of-mass boost, keyed augmentation of the whole pool, / scale,
    concatenated in the cache's base order (one (n_events, P, 4) block per segment)"""
    out = []
    for s in meta["segs"]:
        raw = np.load(paths[s["role"]][s["name"]])
        P = s["P"]
        com = exp._boost_com(raw[:, :4 * P], P)
        aug = keyed_aug.augment(com, seed, keyed_aug.pool_uid(keyed_aug.pool_identity(paths[s["role"]][s["name"]])),
                                np.arange(len(com)))
        out.append(aug / mom_div)
    return out


def _gather_all(src, n, pid, bs):
    moms = []
    for i in range(0, n, bs):
        v = np.arange(i, min(n, i + bs))
        m, _, _, _ = src.gather(v, pid[v])
        moms.append(m)
    return np.concatenate(moms)


def test_key_batches_openers_canary_prune():
    rng = np.random.default_rng(3)
    tmp = tempfile.mkdtemp()
    try:
        pd = os.path.join(tmp, "pools"); os.makedirs(pd)
        paths = _pools(pd, rng)
        root = os.path.join(tmp, "cache")
        exp, d, meta, kdoc = _build(root, paths)
        # key: not the path or mtime, no seed; content and format change it
        pd2 = os.path.join(tmp, "pools_copy"); shutil.copytree(pd, pd2)
        time.sleep(0.01)
        paths2 = {r: {n: p.replace(pd, pd2) for n, p in v.items()} for r, v in paths.items()}
        for v in paths2.values():
            for p in v.values():
                os.utime(p, None)
        assert dm._hash(dm.base_key_doc(exp.cfg, ["pa", "pb"], ["train", "val", "test"], paths2)) == dm._hash(kdoc)
        assert "seed" not in json.dumps(kdoc).replace("particle_order_seed", "")
        a = np.load(paths2["val"]["pb"]); a[-1, -1] *= 1.5; np.save(paths2["val"]["pb"], a)
        assert dm._hash(dm.base_key_doc(exp.cfg, ["pa", "pb"], ["train", "val", "test"], paths2)) != dm._hash(kdoc)
        old = dm.MMAP_VERSION
        dm.MMAP_VERSION = old + 1
        try:
            assert dm._hash(dm.base_key_doc(exp.cfg, ["pa", "pb"], ["train", "val", "test"], paths)) != dm._hash(kdoc)
        finally:
            dm.MMAP_VERSION = old
        # batches: two seeds from one cache, bitwise the in-memory computation, independent of the batch grouping
        dm._check_canary(d, meta, paths)
        N = meta["N"]
        rows = rng.permutation(N).astype(np.int64)
        seg_of_row = np.searchsorted([s["ev0"] for s in meta["segs"]], rows, side="right") - 1
        pid = np.array([["pa", "pb"].index(meta["segs"][k]["name"]) for k in seg_of_row])
        for seed in (42, 43):
            ref = _reference(exp, paths, meta, seed)
            want = np.concatenate([ref[k][rows[i] - meta["segs"][k]["ev0"]] for i, k in enumerate(seg_of_row)])
            src = _source(d, meta, seed, rows)
            for bs in (1, 97, 4096):
                got = _gather_all(src, N, pid, bs)
                assert got.dtype == np.float32 and np.array_equal(got, want.reshape(-1, 4)), (seed, bs)
        assert not np.array_equal(_gather_all(_source(d, meta, 42, rows), N, pid, 512),
                                  _gather_all(_source(d, meta, 43, rows), N, pid, 512))
        # two processes open the one cache at once
        ctx = mp.get_context("fork")
        with ctx.Pool(2) as pool:
            res = pool.starmap(_open_and_read, [(d, rows, pid, 42)] * 2)
        mine = _gather_all(_source(d, meta, 42, rows), N, pid, 256)
        assert all(np.array_equal(r, mine) for r in res)
        # canary: an array overwritten after the build is refused
        p = os.path.join(d, "amp.npy")
        mm = np.load(p, mmap_mode="r+"); mm[-1] += 1.0; mm.flush(); del mm
        try:
            dm._check_canary(d, meta, paths)
            raise RuntimeError("the canary did not refuse an overwritten cache")
        except AssertionError:
            pass
        # prune: least recently used first, never one used within min_idle_s
        dA = os.path.join(root, "kA", "train1"); dB = os.path.join(root, "kB", "train1")
        for x in (dA, dB):
            os.makedirs(x); open(os.path.join(x, "meta.json"), "w").write("{}")
            np.save(os.path.join(x, "big.npy"), np.zeros(1000))
            dm._touch(x)
        os.utime(os.path.join(dA, "last_used"), (time.time() - 5 * 86400,) * 2)
        os.utime(os.path.join(dB, "last_used"), (time.time() - 3 * 86400,) * 2)
        free = shutil.disk_usage(root).free
        assert dm.prune(root, free + 1, keep=d, min_idle_s=4 * 86400) == [dA]
        assert os.path.exists(dB) and os.path.exists(d)
    finally:
        shutil.rmtree(tmp)


def _open_and_read(d, rows, pid, seed):
    meta = json.load(open(os.path.join(d, "meta.json")))
    return _gather_all(_source(d, meta, seed, rows), meta["N"], pid, 256)


def test_fingerprint():
    fp = dm.format_fingerprint()
    want = FINGERPRINT.get(dm.MMAP_VERSION)
    assert want is not None, f"record FINGERPRINT[{dm.MMAP_VERSION}] = {fp!r}"
    assert fp == want, (f"the code that writes the memmap cache changed (fingerprint {fp}): bump MMAP_VERSION in "
                        f"data_mmap.py and record the new fingerprint here")


if __name__ == "__main__":
    test_lorentz_and_keys(); print("keyed_aug: ok")
    test_key_batches_openers_canary_prune(); print("key, batches, openers, canary, prune: ok")
    test_fingerprint(); print("fingerprint: ok")
