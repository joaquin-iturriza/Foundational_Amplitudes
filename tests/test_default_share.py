#!/usr/bin/env python
"""catalog_v4's default-sampled share (mg5_pipeline_final "DEFAULT-SAMPLED SHARE", recipes/gen_catalog_v4_default.py):
the partonic-frame boost, the chunk rows, the MadEvent runs per energy, the pool identity and path, the default share
read from its pools (reuse pool first), the mixture with a default share, and the generated entries and recipes.
CPU only, no MadGraph (the pools are written by hand).

Run:  python tests/test_default_share.py
"""
import json
import os
import sys
import tempfile

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "recipes"))   # gen_catalog_v4_default (test_generated), under pytest too
import mg5_pipeline_final as mg  # noqa: E402
import datagen  # noqa: E402

R = lambda f: os.path.join(ROOT, "recipes", f)


def _boosted_events(n, k, rng):
    """n massless 2 -> k-2 events in the partonic frame, then boosted along z by a random rapidity."""
    sq = rng.uniform(100, 1000, n)
    P = np.concatenate([mg._beams(sq, np.array([21] * k), {}), mg._rambo_massless_batch(sq, k - 2, rng)], axis=1)
    y = rng.uniform(-2, 2, n)
    ch, sh = np.cosh(y)[:, None], np.sinh(y)[:, None]
    B = P.copy()
    B[..., 0], B[..., 3] = ch * P[..., 0] + sh * P[..., 3], sh * P[..., 0] + ch * P[..., 3]
    return P, B, sq


def test_partonic_frame():
    rng = np.random.default_rng(1)
    P, B, sq = _boosted_events(50, 5, rng)
    Q = mg.to_partonic_frame(B)
    assert np.allclose(Q, P, atol=1e-8 * 1000), np.abs(Q - P).max()
    assert np.allclose(mg.to_partonic_frame(P), P, atol=1e-9)        # already at rest: unchanged
    print("[frame] ok")


def test_rows():
    rows = mg.default_share_rows(0.5, [3334, 3333, 3333])
    assert rows == [(0, 1667), (1667, 3334), (3334, 5000)], rows          # cumulative: the unchunked [0, 5000)
    small = mg.default_share_rows(0.25, [3334, 3333, 3333])
    assert all(b - a <= d - c for (a, b), (c, d) in zip(small, rows)) and small[-1][1] < rows[-1][1]
    print("[rows] ok")


def test_lhe_runs():
    pp = next(n for n, e in mg.PROCESSES.items() if n.startswith("pp13v4_") and int(e["lhe"].get("n_runs", 1)) > 1)
    k = int(mg.PROCESSES[pp]["lhe"]["n_runs"])
    runs = mg.lhe_runs(pp, 100, 5000)
    assert [r[0] for r in runs] == [(100 + 7919 * j - 1) % 30000 + 1 for j in range(k)]     # the pp13v4 pools' seeds
    assert all(r[1] == 5000 and r[2] == {} for r in runs) and runs[0][3] == "run_01"
    ee = next(n for n in mg.PROCESSES if n.startswith("ee4_"))
    Es = mg.PROCESSES[ee]["lhe"]["energies"]
    runs = mg.lhe_runs(ee, 100, 2000)
    assert [r[2]["ebeam1"] * 2 for r in runs] == Es and len({r[0] for r in runs}) == len(Es)
    assert all(r[1] == -(-2000 // len(Es)) for r in runs)
    rec = mg.variable_energy_recipe(ee, 13000, 13000, 2000, role="test", seed=42)
    assert rec["mode"] == "lhe_pp" and rec["lhe"]["energies"] == Es
    assert "energies" not in mg.variable_energy_recipe(pp, 13000, 13000, 2000, role="test", seed=42)["lhe"]
    print(f"[runs] ok ({pp}: {k} runs; {ee}: energies {Es})")


def _arm_entry(base, f, src):
    name = f"zz_arm{int(100 * f)}_{base}"
    mg.PROCESSES[name] = dict(mg.PROCESSES[base], sampling={**mg.V4_SAMPLING, "f_default": f, "default_source": src})
    return name


def test_identity_and_path():
    src = next(n for n, e in mg.PROCESSES.items() if n.startswith("dsh_") and e["reuse"].startswith("pp13v4_"))
    base = mg.PROCESSES[src]["parent"]
    mg.PROCESSES[base]["sampling"] = dict(mg.V4_SAMPLING)
    r0 = mg.variable_energy_recipe(base, 30, 6000, 100000, role="train", seed=42)
    assert "default_share" not in r0 and "f_default" not in r0["sampling"]
    assert mg.recipe_output_path(r0, "/D").endswith("_train_smix_amplitudes.npy")
    a, b = _arm_entry(base, 0.5, src), _arm_entry(base, 0.25, src)
    ra, rb = (mg.variable_energy_recipe(n, 30, 6000, 100000, role="train", seed=42) for n in (a, b))
    assert mg.recipe_output_path(ra, "/D").endswith("_train_smixd50_amplitudes.npy")
    assert mg.recipe_output_path(rb, "/D").endswith("_train_smixd25_amplitudes.npy")
    pools = [p["process"] for p in ra["default_share"]["pools"]]
    assert pools == [mg.PROCESSES[src]["reuse"], src], pools
    assert len({mg.recipe_id(r) for r in (r0, ra, rb)}) == 3
    bad = dict(mg.PROCESSES[base], sampling={**mg.V4_SAMPLING, "f_default": 0.5})
    try:
        mg.sampling_policy(bad)
        raise AssertionError("f_default without a source accepted")
    except ValueError:
        pass
    # the chunks of an arm pool read consecutive default rows; a catalog_v4 pool's tasks are unchanged
    t = datagen.chunk_tasks(a, 30, 6000, 100000, "train", 42, "/tmp/x")
    assert [x["default_rows"] for x in t] == [list(r) for r in mg.default_share_rows(0.5, [x["count"] for x in t])]
    assert all("default_rows" not in x for x in datagen.chunk_tasks(base, 30, 6000, 100000, "train", 42, "/tmp/x"))
    for n in (a, b):
        del mg.PROCESSES[n]
    print(f"[identity] ok ({base} <- {src}, {len(t)} chunks)")


def _write_pool(rec, path, P, amps):
    n, k = P.shape[:2]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pdg = np.tile(np.asarray(rec["pdg_ids"], float), (n, 1))
    np.save(path, np.concatenate([P.reshape(n, 4 * k), pdg, amps[:, None]], axis=1))
    with open(path + ".recipe.json", "w") as f:
        json.dump({"recipe_id": mg.recipe_id(rec)}, f)


def test_read_and_mixture():
    """Fake reuse and dsh pools of a 2 -> 3 gluon class (boosted events), read back in the partonic frame, then
    a mixture with the default share labelled by a toy |M|^2."""
    rng = np.random.default_rng(2)
    base = "zz_ds_base"
    k = 5
    mg.PROCESSES[base] = {"pdg_ids": [21] * k, "m_finals": [0.0] * 3, "nfinal": 3, "mg5_generate": ["generate g g > g g g"]}
    lhe = {"cards_dir": "recipes/catalog_v4_pp", "run_card": "run_card.dat", "param_card": "param_card.dat",
           "label_alphas": "running_sqrt_shat"}
    mg.PROCESSES["zz_ds_reuse"] = dict(mg.PROCESSES[base], kind="lhe_pp", lhe={**lhe, "n_val": 0, "n_test": 0, "n_train": 30})
    mg.PROCESSES["zz_ds_src"] = dict(mg.PROCESSES[base], kind="lhe_pp", reuse="zz_ds_reuse",
                                     lhe={**lhe, "n_val": 0, "n_test": 0, "n_train": 50})
    old = datagen.dest_for_role
    with tempfile.TemporaryDirectory() as d:
        datagen.dest_for_role = lambda role: os.path.join(d, role)
        try:
            pools = mg.default_share_pools("zz_ds_src", "train", 42)
            truth = []
            for (name, rec, path), n in zip(pools, (30, 50)):
                P, B, _ = _boosted_events(n, k, rng)
                _write_pool(rec, path, B, np.ones(n))
                truth.append(P)
            truth = np.concatenate(truth)
            P, sq = mg.default_share_momenta("zz_ds_src", "train", 42, (20, 60))   # spans both pools
            assert np.allclose(P, truth[20:60], atol=1e-6)
            # the pool's window: events outside [lo, hi] are dropped before the rows are counted
            sq_all = np.sqrt(np.clip((lambda q: q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1))(truth[:, 0] + truth[:, 1]), 0, None))
            lo, hi = np.quantile(sq_all, [0.2, 0.8])
            Pw, sw = mg.default_share_momenta("zz_ds_src", "train", 42, (0, 10), accept=(lo, hi, [0.0] * 3, [21] * k, None))
            inw = truth[(sq_all >= lo) & (sq_all <= hi)]
            assert np.allclose(Pw, inw[:10], atol=1e-6) and np.all((sw >= lo) & (sw <= hi))
            cuts = dict(mg.V4_CUTS["hadronic"])
            Pc, _ = mg.default_share_momenta("zz_ds_src", "train", 42, (0, 5), accept=(0, 1e9, [0.0] * 3, [21] * k, cuts))
            assert mg.fiducial_pass_mask(Pc, np.zeros(3), np.array([21] * k), cuts).all()
            assert np.allclose(sq, np.sqrt(np.clip((lambda q: q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1))(truth[20:60, 0] + truth[20:60, 1]), 0, None)))
            try:
                mg.default_share_momenta("zz_ds_src", "train", 42, (70, 90))
                raise AssertionError("rows past the pools accepted")
            except RuntimeError:
                pass
            os.remove(pools[0][2] + ".recipe.json")
            try:
                mg.default_share_momenta("zz_ds_src", "train", 42, (0, 10))
                raise AssertionError("a pool without its recipe id accepted")
            except RuntimeError:
                pass
        finally:
            datagen.dest_for_role = old
    # the mixture: 200 events, f_flat 0.35, f_fwd 0.15, 40 default events -> bulk 30
    pol = {**mg.DEFAULT_SAMPLING, **mg.V4_SAMPLING}
    label = lambda ev, s: np.array([1.0 + abs(e[0][2, 3]) / s_ for e, s_ in zip(ev, s)])
    cuts = dict(mg.V4_CUTS["hadronic"])
    D = truth[:40]
    Dsq = np.sqrt(np.clip((lambda q: q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1))(D[:, 0] + D[:, 1]), 0, None))
    ev, s, me = mg.build_mixture_dataset(200, 30, 6000, [0.0] * 3, [21] * k, np.random.default_rng(3), cuts, pol,
                                         label, "log", {}, (D, Dsq))
    assert len(ev) == 200 and len(s) == 200 and len(me) == 200
    M = np.stack([e[0] for e in ev])
    hit = [np.where(np.all(np.isclose(M, D[i], atol=1e-12), axis=(1, 2)))[0] for i in range(40)]
    assert all(len(h) == 1 for h in hit), "a default event missing or repeated"
    idx = np.array([h[0] for h in hit])
    assert np.allclose(me[idx], label([(D[i], None) for i in range(40)], Dsq)) and np.allclose(s[idx], Dsq)
    try:
        mg.build_mixture_dataset(50, 30, 6000, [0.0] * 3, [21] * k, np.random.default_rng(3), cuts, pol, label,
                                 "log", {}, (D, Dsq))
        raise AssertionError("shares over the pool accepted")
    except ValueError:
        pass
    for n in (base, "zz_ds_reuse", "zz_ds_src"):
        del mg.PROCESSES[n]
    print("[read+mixture] ok")


def test_generated():
    """The generated entries and recipes: every training tree class of a hadronic or e+e- beam has a dsh source whose
    events plus its reuse pool's cover the largest arm's share in every role; the arms differ from catalog_v4_train.yaml
    only by the share; the e+e- card is an e+e- card."""
    import gen_catalog_v4_default as g
    base = yaml.safe_load(open(R("catalog_v4_train.yaml")))
    for f in g.ARMS:
        arm = yaml.safe_load(open(R(f"catalog_v4_default{int(round(100 * f)):02d}_train.yaml")))
        assert len(arm["processes"]) == len(base["processes"])
        for a, b in zip(arm["processes"], base["processes"]):
            assert {k: v for k, v in a.items() if k != "sampling"} == {k: v for k, v in b.items() if k != "sampling"}
            if a.get("sampling"):
                assert a["sampling"]["f_default"] == f and a["sampling"]["default_source"] in mg.PROCESSES
    dsh = yaml.safe_load(open(R("catalog_v4_default_processes.yaml")))["processes"]
    for n, e in dsh.items():
        cfg = mg.PROCESSES[n]
        for role, size in (("train", 100000), ("val", 10000), ("test", 10000)):
            have = sum(hi - lo for lo, hi in [mg.lhe_layout(mg.PROCESSES[x])[role] for x in (e["reuse"], n)])
            assert have >= np.ceil(g.F_MAX * size * (1 + g.ACCEPT_MARGIN)), (n, role, have)
        assert cfg["mg5_generate"] == mg.PROCESSES[e["parent"]]["mg5_generate"]
    # every arm pool's chunks ask at most what its default pools hold before acceptance, less the margin
    for f in g.ARMS:
        arm = yaml.safe_load(open(R(f"catalog_v4_default{int(round(100 * f)):02d}_train.yaml")))
        for e in arm["processes"]:
            if not e.get("sampling"):
                continue
            src = e["sampling"]["default_source"]
            mg.PROCESSES[e["name"]]["sampling"] = e["sampling"]
            for role, key in (("train", "n_train"), ("val", "n_val"), ("test", "n_test")):
                t = datagen.chunk_tasks(e["name"], e["sqrts"][0], e["sqrts"][1], int(e[key]), role, 42, "/tmp/x")
                end = t[-1]["default_rows"][1]
                assert end == round(f * int(e[key])), (e["name"], role, end)
                have = sum(hi - lo for lo, hi in [mg.lhe_layout(mg.PROCESSES[x])[role]
                                                  for x in (mg.PROCESSES[src]["reuse"], src)])
                assert end * (1 + g.ACCEPT_MARGIN) <= have + 1, (e["name"], role, end, have)
            del mg.PROCESSES[e["name"]]["sampling"]
    card = open(R("catalog_v4_ee/run_card.dat")).read()
    assert "0\t= lpp1" in card and "none\t= pdlabel" in card and "2.6467\t= etaj" in card and "10.0\t= mmll" in card
    print(f"[generated] ok ({len(dsh)} dsh entries)")


def test_chunked_full_shares():
    """f_default 0.50 fills the pool with 0.35 flat and 0.15 forward (no bulk): chunks of [334, 333, 333] round the
    flat and forward shares per chunk and the default rows cumulatively (default_share_rows), which overshot a chunk by
    one event before the remainder went to the forward share. Every chunk must come out at its count, and the chunks
    together read the default rows [0, round(0.5 N)) exactly once."""
    k = 5
    rng = np.random.default_rng(5)
    counts = [334, 333, 333]
    pol = {**mg.DEFAULT_SAMPLING, **mg.V4_SAMPLING, "f_default": 0.5, "default_source": "zz_unused"}
    label = lambda ev, s: np.array([1.0 + abs(e[0][2, 3]) / s_ for e, s_ in zip(ev, s)])
    cuts = dict(mg.V4_CUTS["hadronic"])
    n_d = int(round(0.5 * sum(counts)))
    D, _, Dsq = _boosted_events(n_d, k, rng)
    seen = []
    for c, (a, b) in zip(counts, mg.default_share_rows(0.5, counts)):
        ev, s, me = mg.build_mixture_dataset(c, 30, 6000, [0.0] * 3, [21] * k, np.random.default_rng(10 + c), cuts,
                                             pol, label, "log", {}, (D[a:b], Dsq[a:b]))
        assert len(ev) == c and len(s) == c and len(me) == c, (c, len(ev))
        M = np.stack([e[0] for e in ev])
        for i in range(a, b):
            assert len(np.where(np.all(np.isclose(M, D[i], atol=1e-12), axis=(1, 2)))[0]) == 1, (c, i)
        seen += list(range(a, b))
    assert seen == list(range(n_d)), "default rows not read exactly once"
    print("[chunked full shares] ok")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(ROOT, "recipes"))
    test_partonic_frame()
    test_rows()
    test_lhe_runs()
    test_identity_and_path()
    test_read_and_mixture()
    test_chunked_full_shares()
    test_generated()
