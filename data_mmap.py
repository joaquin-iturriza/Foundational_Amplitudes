"""data.memmap: the recipe data path with host memory that scales with the batch, not the dataset.

One data cache per pool set, shared by every run on it whatever its seed, caps or statistics (D290, 2026-11-06; the
blind review of 2026-11-02). Two levels under $SCRATCH/amp_mmap_cache (data.memmap_dir overrides the root):

  <key>/<depth>/            the base arrays: every (role, process) pool's first `depth` events, in load order, as
                            they are in the pool before anything a run chooses: the momenta boosted to the centre of
                            mass of the incoming pair (float32, GeV), the raw PDG ids, the raw |M|^2, sqrt(s_hat) and
                            the per-event alpha_s; meta.json holds the per-pool segments and canaries.
  <key>/<depth>/feat_<fk>/  the frame-independent per-event inputs a configuration derives from them: the target
                            factor's log (data.target_physics_factor) and the raw off-shellness columns
                            (offshell_per_event), keyed by what they are computed from.

<key> is the hash of the pools' content identities (keyed_aug.pool_identity: recipe_id, size, the bytes at both
ends; never the path or mtime), the layout (roles, process order, particle_order) and MMAP_VERSION. <depth> is the
largest per-role cap the cache was built for: a run with smaller caps reads a prefix of every pool (as the in-memory
path does with [:cap]), so the train_subsample or eval_subsample of a run never builds a new cache. No seed, frozen
statistic, tokenizer or source text enters either key: the run applies them when it opens the cache (the selection,
the tokens, the amplitude transform and standardization, the off-shellness standardization, all fitted on its own
train selection or taken frozen) and per batch (the random Lorentz augmentation drawn per event from
(seed, pool, index) by keyed_aug, the momentum scale, the tokens, the order features). The in-memory path draws the
same transforms, so both paths give the same batches for a seed.

The cache is built by a CPU prebuild job (scripts/job_mmap_prebuild.sh: the run's own config with
++data.memmap_prebuild=true), never inside a GPU job: a run that finds no cache refuses with the command that builds
it (a build inside a GPU job holds its GPU idle for the build, and concurrent jobs would race to build it; falling back
to the in-memory path would change the host memory the job was sized for). The build takes a lock per depth
(dirlock), checks the free disk against the arrays' size and prunes the least recently used caches of the root when
short, preallocates every file (posix_fallocate), stages in node-local $TMPDIR when it has room (else in a staging dir
on the cache's filesystem) named by host, job and pid, and appears by one rename; it never deletes another job's
staging dir. A run checks the canaries on open (array sizes and the bytes at both ends of every array, and the first
and last |M|^2 of every pool segment against the pool file itself) and touches last_used, which the pruning reads.

Not on this path (asserted): data.target_propagators, training.sign_head, data.resonance_per_event,
data.phase_space_on_pool, data.target_physics_hybrid.
"""
import socket
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

import logging.handlers  # noqa: F401  (logger.py uses logging.handlers without importing it)

from logger import LOGGER
from dirlock import dir_lock
import keyed_aug
from keyed_aug import pool_identity

# The format of the cached arrays. Bump it with any change to what _build_base or _build_features write (the
# fingerprint in tests/test_mmap_key.py fails until it is bumped and re-recorded).
MMAP_VERSION = 3


def format_fingerprint():
    """sha256 of the code that writes the cached arrays (the base and derived builds, the centre-of-mass boost, the
    target factor and the off-shellness masks): tests/test_mmap_key.py pins it to MMAP_VERSION, so a change to what
    the cache holds fails the test until the version is bumped. The key itself never hashes source text."""
    import inspect
    import experiment
    E = experiment.AmplitudeExperiment
    me = sys.modules[__name__]
    parts = [inspect.getsource(getattr(me, n)) for n in ("_build_base", "_build_features", "_alloc")]
    parts += [inspect.getsource(f) for f in (E._boost_com, E._physics_logF, E._eikonal_logS, E._setup_offshell_masks)]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()
CHUNK = 1 << 22            # elements per streamed chunk (a multiple of the 8192-element reduction block)
NP_BLOCK = 8192            # numpy's pairwise-sum block over a contiguous float reduction (NPY_BUFSIZE)


# ----------------------------------------------------------------------------------------------------------------
# Streamed reductions with numpy's own summation order
# ----------------------------------------------------------------------------------------------------------------
class F32FlatSum:
    """np.add.reduce over a contiguous float32 array fed in pieces, bitwise: numpy sums a contiguous reduction in
    blocks of 8192 elements, each block pairwise, the block sums accumulated one after the other in float32."""

    def __init__(self):
        self.acc = None
        self.buf = np.empty(0, np.float32)
        self.n = 0

    def _block(self, s):
        self.acc = s if self.acc is None else np.float32(self.acc + s)

    def add(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        self.n += x.size
        if self.buf.size:
            take = min(NP_BLOCK - self.buf.size, x.size)
            self.buf = np.concatenate([self.buf, x[:take]])
            x = x[take:]
            if self.buf.size == NP_BLOCK:
                self._block(np.add.reduce(self.buf)); self.buf = np.empty(0, np.float32)
        k = x.size // NP_BLOCK
        if k:
            for s in np.add.reduce(x[:k * NP_BLOCK].reshape(k, NP_BLOCK), axis=1):
                self._block(s)
        if x.size - k * NP_BLOCK:
            self.buf = x[k * NP_BLOCK:].copy()

    def result(self):
        if self.buf.size:
            self._block(np.add.reduce(self.buf)); self.buf = np.empty(0, np.float32)
        return np.float32(0.0) if self.acc is None else self.acc


def f32_mean_std(chunks, scale=None):
    """(np.mean(a), np.std(a)) of the float32 array a = concatenation of chunks() (or of chunks() / scale), bitwise
    numpy's _mean / _var / _std on the concatenated array. `chunks` is a callable returning a fresh iterator of float32
    pieces (it is read twice: the mean, then the squared deviations)."""
    def pieces():
        for c in chunks():
            yield c if scale is None else c / scale
    s = F32FlatSum()
    for c in pieces():
        s.add(c)
    n = np.intp(s.n)
    tot = s.result()
    mean = tot.dtype.type(tot / n)                                             # _mean: numpy scalar branch
    arrmean = np.array(tot, dtype=np.float32).reshape(1, 1)                    # _var: keepdims ndarray branch
    arrmean = np.true_divide(arrmean, n, out=arrmean, casting="unsafe", subok=False)
    q = F32FlatSum()
    for c in pieces():
        x = np.asanyarray(c - arrmean)
        x = np.multiply(x, x, out=x)
        q.add(x)
    ret = q.result()
    ret = ret.dtype.type(ret / np.maximum(n - 0, 0))
    std = ret.dtype.type(np.sqrt(ret))
    return mean, std


def _rowsum(it):
    """Sequential axis-0 sum of the row blocks of `it` (numpy adds the rows of a C-order (N, C) array one after
    the other: row 0, then += row 1, ...); returns (sum, number of rows)."""
    acc, n = None, 0
    for c in it:
        n += c.shape[0]
        acc = np.add.reduce(c if acc is None else np.concatenate([acc[None], c]), axis=0)
    return acc, n


def rows_mean_std(chunks):
    """(a.mean(0), a.std(0)) of the float64 (N, C) array a = concatenation of chunks() (a callable returning a fresh
    iterator of row blocks; read twice), bitwise numpy's _mean / _var / _std."""
    S, n = _rowsum(chunks())
    rc = np.intp(n)
    mean = np.true_divide(S, rc, casting="unsafe", subok=False)
    arrmean = np.true_divide(S.reshape(1, -1), rc, casting="unsafe", subok=False)

    def dev():
        for c in chunks():
            x = np.asanyarray(c - arrmean)
            yield np.multiply(x, x, out=x)
    Q, _ = _rowsum(dev())
    var = np.true_divide(Q, np.maximum(rc - 0, 0), casting="unsafe", subok=False)
    return mean, np.sqrt(var)


# ----------------------------------------------------------------------------------------------------------------
# Dataset and sampler over the memory-mapped arrays


# ----------------------------------------------------------------------------------------------------------------
# The run's view of the cache: what a batch is gathered and computed from
# ----------------------------------------------------------------------------------------------------------------
class MmapSource:
    """Everything a batch needs besides its rows: the base maps (momenta, PDG ids, sqrt(s_hat), each event's first
    particle row), the run's event map (virtual row -> base event row), the per-segment keys of the augmentation, the
    token table, the order rows, the off-shellness map and its standardization, the momentum scale and the seed."""

    def __init__(self, d, fdir, meta, ev_rows, seed, mom_div, dimless, tok_U, tok_lut, order_rows,
                 osh_stats, osh_eps, F):
        self.mom, self.pdg, self.rs, self.evp = (_open(d, n) for n in ("mom", "pdg", "rs", "evp"))
        segs = meta["segs"]
        self.seg_ev0 = np.array([s["ev0"] for s in segs], dtype=np.int64)
        self.seg_P = np.array([s["P"] for s in segs], dtype=np.int64)
        self.seg_uid = np.array([s["uid"] for s in segs], dtype=np.uint64)
        self.ev_rows = ev_rows
        self.seed, self.mom_div, self.dimless = int(seed), float(mom_div), bool(dimless)
        self.tok_U, self.tok_lut, self.order_rows = tok_U, tok_lut, order_rows
        self.osh = _open(fdir, "osh") if osh_stats is not None else None
        self.osh_mu, self.osh_sd = ((np.asarray(osh_stats[0], np.float64), np.asarray(osh_stats[1], np.float64))
                                    if osh_stats is not None else (None, None))
        self.osh_eps, self.F = osh_eps, int(F)

    def osh_transformed(self, base_rows):
        """sign(o) log1p|o| of the raw off-shellness of `base_rows` (o / eps under dimensionless_inputs), float64"""
        o = np.asarray(self.osh[base_rows]).astype(np.float64)
        if self.dimless:
            o = o / self.osh_eps
        return np.sign(o) * np.log1p(np.abs(o))

    def gather(self, vrows, pid):
        """(momenta (m, 4) float32, tokens (m,) int64, order (n, F[+1]) float32, counts (n,)) of the virtual rows"""
        r = self.ev_rows[vrows]
        seg = np.searchsorted(self.seg_ev0, r, side="right") - 1
        P = self.seg_P[seg]
        csum = np.cumsum(P)
        total = int(csum[-1]) if len(csum) else 0
        gather = np.repeat(np.asarray(self.evp[r]) - (csum - P), P) + np.arange(total, dtype=np.int64)
        T = keyed_aug.trafo_components(self.seed, self.seg_uid[seg], r - self.seg_ev0[seg])
        ev_of = np.repeat(np.arange(len(r)), P)
        T = [[t[ev_of] for t in row] for row in T]
        mom = keyed_aug.apply_components(T, np.asarray(self.mom[gather])).astype(np.float32)
        if self.dimless:
            rs_p = np.repeat(np.asarray(self.rs[r]), P)
            mom = (mom.astype(np.float64) / rs_p[:, None]).astype(np.float32)
        mom = mom / self.mom_div
        pdg = np.asarray(self.pdg[gather])
        tok = self.tok_lut[np.repeat(pid, P), np.searchsorted(self.tok_U, pdg)]
        order = self.order_rows[pid]
        if self.osh is not None:
            n_pdg = self.osh.shape[1]
            order[:, self.F - n_pdg:self.F] = ((self.osh_transformed(r) - self.osh_mu) / self.osh_sd).astype(np.float32)
        if self.dimless:
            lrs = np.log10(np.asarray(self.rs[r])).astype(np.float32)
            order = np.concatenate([order, lrs[:, None]], axis=1)
        return mom, tok, order, P


class MmapAmplitudeDataset(Dataset):
    """AmplitudeDataset over the run's view of the cache (MmapSource), restricted to `rows` (a range, never
    materialised, or an int64 array of virtual rows). A batch is gathered from the maps (numpy reads only the pages it
    touches; the pages are the file's, shared with every dataloader worker), augmented and scaled, and returned
    exactly as AmplitudeDataset.__getitems__ returns it, the integers widened to int64 and the floats cast to
    `dtype`."""

    def __init__(self, src, amplitudes, process_ids, dtype, rows):
        self.src, self.amplitudes, self.process_ids, self.dtype = src, amplitudes, process_ids, dtype
        self.rows = rows if isinstance(rows, range) else np.asarray(rows, dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def _rows(self, idx):
        if isinstance(self.rows, range):
            assert self.rows.step == 1
            return idx + self.rows.start
        return self.rows[idx]

    def _f(self, a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(self.dtype)

    def __getitems__(self, indices):
        from dataset import _Prebatched
        if os.environ.get("LLOCA_FETCH", "batch") == "item":
            return [self[int(i)] for i in indices]
        v = self._rows(np.asarray(indices, dtype=np.int64))
        pid = np.asarray(self.process_ids[v]).astype(np.int64)
        mom, tok, order, P = self.src.gather(v, pid)
        ptr = torch.zeros(len(v) + 1, dtype=torch.long)
        ptr[1:] = torch.from_numpy(np.cumsum(P))
        return [_Prebatched((
            self._f(mom),
            self._f(np.asarray(self.amplitudes[v])),
            torch.from_numpy(tok.astype(np.int64)),
            self._f(order),
            ptr,
            torch.from_numpy(pid),
        ))]

    def __getitem__(self, i):
        v = np.array([int(self._rows(np.int64(i)))], dtype=np.int64)
        pid = np.asarray(self.process_ids[v]).astype(np.int64)
        mom, tok, order, _ = self.src.gather(v, pid)
        return (
            self._f(mom),
            self._f(np.asarray(self.amplitudes[v[0]])),
            torch.from_numpy(tok.astype(np.int64)),
            self._f(order[0]),
            torch.tensor(int(pid[0]), dtype=torch.long),
        )


class RandomBatchSampler(Sampler):
    """BatchSampler(RandomSampler(range(n)), batch_size, drop_last) without RandomSampler's per-epoch
    randperm(n).tolist(): a Python list of n ints (~36 B per event, 4.9 GB at 1.35e8 train events) in the main
    process. Same draws (the seed from the global torch RNG at the first batch of an epoch, then a private
    generator's randperm), same batches in the same order; the batches are slices of the permutation (8 B per
    event)."""

    def __init__(self, n, batch_size, drop_last=True):
        self.n, self.batch_size, self.drop_last = int(n), int(batch_size), bool(drop_last)

    def __len__(self):
        return self.n // self.batch_size if self.drop_last else -(-self.n // self.batch_size)

    def __iter__(self):
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(self.n, generator=g).numpy()
        stop = len(self) * self.batch_size if self.drop_last else self.n
        for i in range(0, stop, self.batch_size):
            yield perm[i:i + self.batch_size]


def first_k_per_process(process_ids, start, stop, n_proc, k):
    """{p: rows of process p among rows [start, stop) of `process_ids`, ascending, the first k of them (all if k is
    None)}, i.e. split_idx[pid == p][:k] for split_idx = arange(start, stop), read in chunks and stopped as soon as
    every process has k."""
    out = {p: [] for p in range(n_proc)}
    need = np.full(n_proc, np.iinfo(np.int64).max if k is None else int(k), dtype=np.int64)
    pos = start
    while pos < stop and need.max() > 0:
        end = min(stop, pos + CHUNK)
        pid = np.asarray(process_ids[pos:end]).astype(np.int64)
        order = np.argsort(pid, kind="stable")
        cnt = np.bincount(pid, minlength=n_proc)
        bounds = np.concatenate([[0], np.cumsum(cnt)])
        for p in np.flatnonzero((cnt > 0) & (need > 0)):
            sel = order[bounds[p]:bounds[p + 1]][:need[p]] + pos
            out[p].append(sel)
            need[p] -= len(sel)
        pos = end
    return {p: (np.concatenate(v).astype(np.int64) if v else np.empty(0, np.int64)) for p, v in out.items()}


# ----------------------------------------------------------------------------------------------------------------
# Cache build / open


# ----------------------------------------------------------------------------------------------------------------
# Keys
# ----------------------------------------------------------------------------------------------------------------
def _alphas_path(pool):
    """the per-event alpha_s next to a pool (mg5_pipeline_final.build_lhe_pp; tools/recover_alphas.py)"""
    return pool[:-len(".npy")] + ".alphas.npy"


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return repr(float(x))
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _small_file_id(path):
    """[basename, sha256 of the content] of a small input file (a diagram sidecar, the coupling table), or None"""
    if path is None or not os.path.exists(path):
        return None
    return [os.path.basename(path), hashlib.sha256(open(path, "rb").read()).hexdigest()]


def _file_digest(exp, path):
    """sha256 of a file the cached targets depend on (the coupling table of data.target_physics_factor), resolved as
    experiment._physics_logF resolves it"""
    p = os.path.expandvars(str(path))
    p = p if os.path.isabs(p) else os.path.join(os.path.dirname(os.path.abspath(experiment_file())), p)
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def experiment_file():
    import experiment
    return experiment.__file__


def _hash(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:24]


def base_key_doc(cfg, names, roles, paths):
    """What the base arrays are made of: the pools' content, their layout and the format. Nothing a run chooses."""
    porder = str(cfg.data.get("particle_order", "none"))
    return {
        "format": MMAP_VERSION,
        "roles": list(roles),
        "names": list(names),
        "pools": {r: [pool_identity(paths[r][n]) for n in names] for r in roles},
        "alphas": {r: [pool_identity(a) if os.path.exists(a) else None
                       for a in (_alphas_path(paths[r][n]) for n in names)] for r in roles},
        "particle_order": porder,
        # particle_order=shuffle draws the slot order from data.seed (a data choice, not the run's seed)
        **({"particle_order_seed": int(cfg.data.get("seed", 42))} if porder == "shuffle" else {}),
    }


def _orders_inputs(exp, names):
    import mg5_pipeline_final as mg
    ddir = exp.cfg.model.get("diagrams_dir", "data/diagrams")
    if not os.path.isabs(ddir):
        ddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ddir)
    out = []
    for n in names:
        pc = mg.PROCESSES.get(n, {})
        out.append(_jsonable([list(exp._orders_of(n)),
                              {k: pc.get(k) for k in ("kind", "order", "alphas_prefactor", "nfinal")},
                              exp._diagram_born_orders(n), _small_file_id(exp._resolve_diagram_path(ddir, n))]))
    return out


def features_key_doc(exp, names, want_offshell, sidecars):
    """What the derived per-event inputs are computed from (besides the base arrays): the target factor's
    definition and the off-shellness masks' inputs. Their standardization is the run's, at open."""
    cfg = exp.cfg
    phys_on = bool(cfg.data.get("target_physics_factor", False))
    orders_on = bool(cfg.data.get("target_physics_orders", False))
    return _jsonable({
        "format": MMAP_VERSION,
        "phys": ([float(cfg.data.get("target_physics_s0", 289.0)),
                  str(cfg.data.get("target_physics_couplings", "default")),
                  _file_digest(exp, cfg.data.get("target_physics_couplings",
                                                 "analysis/catalog_daniel/data/diagram_couplings.json")),
                  bool(cfg.data.get("target_physics_dim_only", False)),
                  str(cfg.data.get("target_physics_dim_ref", "shat")),
                  bool(cfg.data.get("target_physics_eikonal", False)),
                  float(cfg.data.get("target_physics_qperp_tau", 0.3)),
                  float(cfg.data.get("target_physics_alpha_inv", 132.507)),
                  orders_on, exp._target_orders_source() if orders_on else None]
                 if phys_on else None),
        "orders_in": _orders_inputs(exp, names) if orders_on else None,
        "offshell": ({
            "pdgs": exp._internal_mass_pdgs,
            "sidecars": [_small_file_id(p) for p in sidecars],
            "internal_masses": getattr(exp, "_internal_mass_by_proc", None),
            "per_shat": bool(cfg.data.get("dimensionless_inputs", False)),
        } if want_offshell else None),
    })


# ----------------------------------------------------------------------------------------------------------------
# Cache directories: depth, disk, staging, pruning, canaries
# ----------------------------------------------------------------------------------------------------------------
def _cache_root(cfg):
    root = cfg.data.get("memmap_dir", None)
    if root in (None, "null", "none", ""):
        root = os.path.join(os.environ.get("SCRATCH") or os.path.join(cfg.run_dir, ".."), "amp_mmap_cache")
    return os.path.abspath(str(root))


def _open(d, name):
    return np.load(os.path.join(d, name + ".npy"), mmap_mode="r")


def _cap(c):
    return int(c) if c not in (None, "null", "none") else None


def _need(cap_for, names, roles, paths):
    """{(role, name): events the run reads from the pool (its first ones)}"""
    out = {}
    for r in roles:
        cap = _cap(cap_for[r])
        for n in names:
            N0 = np.load(paths[r][n], mmap_mode="r").shape[0]
            out[(r, n)] = N0 if cap is None else min(N0, cap)
    return out


def _depth_name(cap_for, roles):
    return "_".join(f"{r}{_cap(cap_for[r]) if _cap(cap_for[r]) is not None else 'all'}" for r in roles)


def _find_depth(kdir, need):
    """The smallest built depth under kdir whose every segment holds at least the events the run reads, or None"""
    best = None
    if not os.path.isdir(kdir):
        return None
    for dn in sorted(os.listdir(kdir)):
        mp = os.path.join(kdir, dn, "meta.json")
        if dn.startswith(".") or not os.path.exists(mp):
            continue
        meta = json.load(open(mp))
        if all(s["n"] >= need[(s["role"], s["name"])] for s in meta["segs"]):
            tot = sum(s["n"] for s in meta["segs"])
            if best is None or tot < best[0]:
                best = (tot, os.path.join(kdir, dn), meta)
    return None if best is None else best[1:]


def _canary(d, arrays):
    """{array: [size in bytes, sha256 of its first and last 64 KiB]} of the files in d"""
    out = {}
    for a in arrays:
        p = os.path.join(d, a + ".npy")
        size = os.path.getsize(p)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            h.update(f.read(1 << 16))
            f.seek(max(0, size - (1 << 16)))
            h.update(f.read(1 << 16))
        out[a] = [int(size), h.hexdigest()]
    return out


def _check_canary(d, meta, paths):
    now = _canary(d, list(meta["canary"]))
    bad = [a for a in meta["canary"] if now[a] != meta["canary"][a]]
    assert not bad, f"memmap cache {d}: {bad} differ from what the build wrote (truncated or overwritten); rebuild it"
    amp = _open(d, "amp")
    for s in meta["segs"]:
        if s["n"] == 0:
            continue
        pool = np.load(paths[s["role"]][s["name"]], mmap_mode="r")
        first, last = float(pool[0, -1]), float(pool[s["n"] - 1, -1])
        assert first == float(amp[s["ev0"]]) and last == float(amp[s["ev0"] + s["n"] - 1]), \
            f"memmap cache {d}: segment {s['role']}/{s['name']} does not match its pool {paths[s['role']][s['name']]}"


def _touch(d):
    p = os.path.join(d, "last_used")
    with open(p, "a"):
        pass
    os.utime(p, None)


def _last_used(d):
    p = os.path.join(d, "last_used")
    return os.path.getmtime(p) if os.path.exists(p) else os.path.getmtime(d)


def _du(d):
    return sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(d) for f in fs)


def prune(root, need_bytes, keep, min_idle_s):
    """Remove the least recently used built depths under root (never `keep`, never one used within min_idle_s, never
    a staging dir or a lock) until need_bytes are free on root's filesystem. Returns the removed dirs."""
    cands = []
    for k in os.listdir(root) if os.path.isdir(root) else []:
        kd = os.path.join(root, k)
        if not os.path.isdir(kd) or k.startswith("."):
            continue
        for dn in os.listdir(kd):
            d = os.path.join(kd, dn)
            if (dn.startswith(".") or dn.endswith(".lockdir") or not os.path.exists(os.path.join(d, "meta.json"))
                    or os.path.abspath(d) == os.path.abspath(keep)):
                continue
            cands.append((_last_used(d), d))
    removed = []
    for t, d in sorted(cands):
        if shutil.disk_usage(root).free >= need_bytes:
            break
        if time.time() - t < min_idle_s:
            continue
        LOGGER.info(f"data.memmap: pruning {d} (last used {time.ctime(t)}, {_du(d) / 1e9:.1f} GB)")
        gone = d + f".pruned.{socket.gethostname()}.{os.getpid()}"
        os.rename(d, gone)              # out of every opener's search at once; open maps keep their pages
        shutil.rmtree(gone, ignore_errors=True)
        removed.append(d)
    return removed


def _job_tag():
    job = os.environ.get("SLURM_JOB_ID") or os.environ.get("_CONDOR_JOB_AD", "").split("/")[-1] or "nojob"
    return f"{socket.gethostname()}.{job}.{os.getpid()}"


def _alloc(path, dtype, shape):
    """np.lib.format.open_memmap(w+) with the file's blocks allocated up front (posix_fallocate), so a full disk
    fails here and not halfway through the build"""
    a = np.lib.format.open_memmap(path, "w+", dtype, shape)
    try:
        fd = os.open(path, os.O_RDWR)
        try:
            os.posix_fallocate(fd, 0, os.path.getsize(path))
        finally:
            os.close(fd)
    except (OSError, AttributeError) as e:   # a filesystem without fallocate: the disk check above still holds
        LOGGER.info(f"data.memmap: posix_fallocate not available on {path} ({e}); relying on the disk check")
    return a


def _staging(final_parent, need_bytes):
    """A new staging dir for one build: in node-local $TMPDIR when it has room for the arrays and is another
    filesystem, else beside the final dir; named by host, job and pid, so no two builds share one"""
    tag = _job_tag()
    tmp = os.environ.get("TMPDIR")
    if tmp and os.path.isdir(tmp) and os.stat(tmp).st_dev != os.stat(final_parent).st_dev \
            and shutil.disk_usage(tmp).free > 1.2 * need_bytes:
        d = os.path.join(tmp, f"amp_mmap_stage.{tag}")
    else:
        d = os.path.join(final_parent, f".stage.{tag}")
    os.makedirs(d)
    return d


def _publish(stage, final):
    """Move a finished staging dir to `final` with one rename on the cache's filesystem"""
    parent = os.path.dirname(final)
    if os.stat(stage).st_dev != os.stat(parent).st_dev:
        local = stage
        stage = os.path.join(parent, ".stage." + _job_tag() + ".copy")
        shutil.copytree(local, stage)
        shutil.rmtree(local, ignore_errors=True)
    os.rename(stage, final)


# ----------------------------------------------------------------------------------------------------------------
# Build (CPU prebuild job only)
# ----------------------------------------------------------------------------------------------------------------
def _base_bytes(need, names, roles, paths):
    M = N = 0
    for r in roles:
        for n in names:
            ncol = np.load(paths[r][n], mmap_mode="r").shape[1]
            P = (ncol - 1) // 5
            N += need[(r, n)]; M += need[(r, n)] * P
    return M * (16 + 4) + N * (8 * 4), M, N


def _build_base(exp, c, d, need, key_doc):
    """Write the base arrays of the pools' first need[(role, name)] events into d; returns the meta"""
    from experiment import order_identical_particles
    cfg = exp.cfg
    names, roles, paths = c["names"], c["roles"], c["paths"]
    porder = key_doc["particle_order"]
    porder_rng = np.random.default_rng(int(cfg.data.get("seed", 42))) if porder == "shuffle" else None
    _, M, N = _base_bytes(need, names, roles, paths)
    mom = _alloc(os.path.join(d, "mom.npy"), np.float32, (M, 4))
    pdg = _alloc(os.path.join(d, "pdg.npy"), np.int32, (M,))
    amp = _alloc(os.path.join(d, "amp.npy"), np.float64, (N,))
    rs = _alloc(os.path.join(d, "rs.npy"), np.float64, (N,))
    evp = _alloc(os.path.join(d, "evp.npy"), np.int64, (N,))
    has_alphas = any(a is not None for r in roles for a in key_doc["alphas"][r])
    alp = _alloc(os.path.join(d, "alphas.npy"), np.float64, (N,)) if has_alphas else None
    segs, pb, eb = [], 0, 0
    for ri, r in enumerate(roles):
        for ni, n in enumerate(names):
            k = need[(r, n)]
            src = np.load(paths[r][n], mmap_mode="r")
            P = (src.shape[1] - 1) // 5
            raw = np.asarray(src[:k])
            parts, pdgs = raw[:, :4 * P], raw[:, 4 * P:-1].astype(int)
            if porder != "none":
                parts = order_identical_particles(parts, pdgs, porder, rng=porder_rng)
            com = exp._boost_com(parts, P)
            cm = com.astype(np.float64)
            q = cm[:, 0] + cm[:, 1]
            mom[pb:pb + k * P] = com.reshape(k * P, 4)
            pdg[pb:pb + k * P] = pdgs.reshape(k * P)
            amp[eb:eb + k] = raw[:, -1]
            rs[eb:eb + k] = np.sqrt(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(axis=1))
            evp[eb:eb + k] = pb + np.arange(k, dtype=np.int64) * P
            if alp is not None:
                ap = _alphas_path(paths[r][n])
                alp[eb:eb + k] = (np.asarray(np.load(ap, mmap_mode="r")[:k], dtype=np.float64)
                                  if key_doc["alphas"][r][ni] is not None else np.nan)
            u, first = np.unique(pdgs.reshape(-1), return_index=True)
            segs.append({"role": r, "name": n, "ev0": eb, "n": int(k), "p0": pb, "P": int(P),
                         "uid": keyed_aug.pool_uid(key_doc["pools"][r][ni]),
                         "pdg_first": [[int(a), int(b) // P] for a, b in zip(u, first)],
                         "slot": [int(x) for x in pdgs[0]] if k else []})
            LOGGER.info(f"  [{r}] {n}: {k} events -> memmap base")
            pb += k * P; eb += k
            del raw, parts, com, cm
    for a in (mom, pdg, amp, rs, evp) + ((alp,) if alp is not None else ()):
        a.flush()
    arrays = ["mom", "pdg", "amp", "rs", "evp"] + (["alphas"] if alp is not None else [])
    return {"key_doc": key_doc, "segs": segs, "M": int(M), "N": int(N), "has_alphas": bool(alp is not None),
            "canary": _canary(d, arrays), "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "by": _job_tag()}


def _build_features(exp, c, d, meta, fd, fkey_doc, want_offshell):
    """Write the frame-independent derived inputs of every stored event into fd (logF, osh); returns their meta"""
    cfg = exp.cfg
    names = c["names"]
    phys_on = fkey_doc["phys"] is not None
    orders_on = bool(cfg.data.get("target_physics_orders", False))
    mom, rs = _open(d, "mom"), _open(d, "rs")
    alp = _open(d, "alphas") if meta["has_alphas"] else None
    N = meta["N"]
    logF = _alloc(os.path.join(fd, "logF.npy"), np.float64, (N,)) if phys_on else None
    n_pdg = len(exp._internal_mass_pdgs) if want_offshell else 0
    osh = _alloc(os.path.join(fd, "osh.npy"), np.float32, (N, n_pdg)) if want_offshell else None
    masks = exp._setup_offshell_masks(names) if want_offshell else None
    pidx = {n: i for i, n in enumerate(names)}
    dimless = bool(cfg.data.get("dimensionless_inputs", False))
    for s in meta["segs"]:
        k, P, e0, p0, name = s["n"], s["P"], s["ev0"], s["p0"], s["name"]
        if k == 0:
            continue
        pp = np.asarray(mom[p0:p0 + k * P]).reshape(k, P, 4).astype(np.float64)
        if phys_on:
            a_s = None
            if orders_on:
                a_s = np.asarray(alp[e0:e0 + k]) if alp is not None else np.full(k, np.nan)
                if np.isnan(a_s).all() and int(exp._orders_of(name)[2]) == 0:
                    a_s = np.ones(k)     # alpha_s power 0: the factor does not depend on alpha_s
                assert not np.isnan(a_s).any(), f"data.target_physics_orders: no per-event alpha_s for {s['role']}/{name}"
            logF[e0:e0 + k] = exp._physics_logF(name, pp, alphas=a_s)
        if want_offshell:
            cols = np.zeros((k, n_pdg), dtype=np.float64)
            sel = masks[pidx[name]] if masks is not None else None
            if sel is not None:
                s_ev = np.asarray(rs[e0:e0 + k]) ** 2 if dimless else 1.0
                for col, pdg in enumerate(exp._internal_mass_pdgs):
                    props = sel.get(pdg, [])
                    if not props:
                        continue
                    offs = []
                    for mrow, mm in props:
                        q = np.einsum("s,nsc->nc", mrow, pp)
                        s_prop = q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(axis=1)
                        offs.append((s_prop - mm) / s_ev)
                    offs = np.stack(offs, axis=1)
                    idx = np.argmin(np.abs(offs), axis=1)
                    cols[:, col] = offs[np.arange(k), idx]
            osh[e0:e0 + k] = cols.astype(np.float32)
        del pp
    arrays = (["logF"] if phys_on else []) + (["osh"] if want_offshell else [])
    for a in (logF, osh):
        if a is not None:
            a.flush()
    return {"key_doc": fkey_doc, "canary": _canary(fd, arrays), "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "by": _job_tag()}


def _ordered_map(fn, items, workers=None, ahead=2):
    """fn over items, yielded in order, computed by `workers` threads at most workers * ahead items ahead"""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    workers = workers or max(1, min(8, len(os.sched_getaffinity(0))))
    with ThreadPoolExecutor(workers) as ex:
        q = deque()
        it = iter(items)
        for x in it:
            q.append(ex.submit(fn, x))
            if len(q) >= workers * ahead:
                yield q.popleft().result()
        while q:
            yield q.popleft().result()


def _locked_build(final, need_bytes, root, build):
    """Build `final` once: under its lock, if no other job published it meanwhile; disk checked (pruning the least
    recently used caches when short), staged, published by a rename. build(stage_dir) writes the arrays and returns
    the meta."""
    parent = os.path.dirname(final)
    os.makedirs(parent, exist_ok=True)
    with dir_lock(final, on_wait=lambda o: LOGGER.info(f"data.memmap: build of {final} held by {o}, waiting")):
        if os.path.exists(os.path.join(final, "meta.json")):
            return json.load(open(os.path.join(final, "meta.json")))
        want = int(1.05 * need_bytes)
        if shutil.disk_usage(parent).free < want:
            prune(root, want, keep=final, min_idle_s=48 * 3600)
        free = shutil.disk_usage(parent).free
        if free < want:
            raise RuntimeError(f"data.memmap: {need_bytes / 1e9:.1f} GB needed for {final}, {free / 1e9:.1f} GB free "
                               f"on its filesystem after pruning the caches idle for 48 h")
        t0 = time.time()
        stage = _staging(parent, need_bytes)
        meta = build(stage)
        with open(os.path.join(stage, "meta.json"), "w") as f:
            json.dump(meta, f)
        _publish(stage, final)
        LOGGER.info(f"data.memmap: built {final} in {time.time() - t0:.0f} s")
        return meta


# ----------------------------------------------------------------------------------------------------------------
# Entry: find (or, in a prebuild job, build) the cache, then open it for this run
# ----------------------------------------------------------------------------------------------------------------
def init_data_recipes_mmap(exp, c):
    """The memory-mapped twin of the tail of AmplitudeExperiment._init_data_recipes (from the per-role loads on).
    `c` holds the locals that function computed before: specs, names, roles, paths, use_pids, per_dataset, the
    frozen statistics (None when fitted here), amp_orders, n_order_features, cap_for, stats_path, stats_on_pool."""
    cfg = exp.cfg
    names, roles, paths = c["names"], c["roles"], c["paths"]
    assert not c["stats_on_pool"], "data.memmap: data.phase_space_on_pool is not implemented on the memmap path"
    assert not bool(cfg.data.get("target_physics_hybrid", False)), \
        "data.memmap: data.target_physics_hybrid (a per-process constant from its events) is implemented on the in-memory path only"
    for k in ("target_propagators", "resonance_per_event"):
        assert not bool(cfg.data.get(k, False)), f"data.memmap: data.{k} is not implemented on the memmap path"
    assert not bool(cfg.training.get("sign_head", False)), "data.memmap: training.sign_head is not implemented"
    phys_on = bool(cfg.data.get("target_physics_factor", False))
    orders_on = bool(cfg.data.get("target_physics_orders", False))
    assert phys_on or not orders_on, \
        "data.target_physics_orders acts through the target factor: set data.target_physics_factor=true"
    exp._target_factor_on = phys_on
    exp._sign_head = False
    want_offshell = (exp._offshell_per_event and exp._use_internal_mass_scalars and bool(exp._internal_mass_pdgs))
    prebuild = bool(cfg.data.get("memmap_prebuild", False))

    root = _cache_root(cfg)
    kdoc = base_key_doc(cfg, names, roles, paths)
    kdir = os.path.join(root, _hash(kdoc))
    need = _need(c["cap_for"], names, roles, paths)
    found = _find_depth(kdir, need)
    rebuild = ("PYTHONPATH=. python run.py <this run's overrides> ++data.memmap=true ++data.memmap_prebuild=true "
               "(scripts/job_mmap_prebuild.sh)")
    if found is None:
        if not prebuild:
            raise RuntimeError(f"data.memmap: no cache of this run's pools at {kdir} deep enough for its caps; build "
                               f"it with a CPU prebuild job first: {rebuild}")
        d = os.path.join(kdir, _depth_name(c["cap_for"], roles))
        nb = _base_bytes(need, names, roles, paths)[0]
        meta = _locked_build(d, nb, root, lambda st: _build_base(exp, c, st, need, kdoc))
        with open(os.path.join(kdir, "key_doc.json"), "w") as f:
            json.dump(kdoc, f, indent=1)
    else:
        d, meta = found
    # the slot order of every process (the off-shellness masks and the eikonal read it)
    exp._slot_pdgs_by_name = {**{s["name"]: s["slot"] for s in meta["segs"] if s["slot"]},
                              **getattr(exp, "_slot_pdgs_by_name", {})}
    ddir = cfg.model.get("diagrams_dir", "data/diagrams")
    if not os.path.isabs(ddir):
        ddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ddir)
    sidecars = [exp._resolve_diagram_path(ddir, n) for n in names] if want_offshell else []
    fdir = None
    if phys_on or want_offshell:
        fkd = features_key_doc(exp, names, want_offshell, sidecars)
        fdir = os.path.join(d, "feat_" + _hash(fkd))
        if not os.path.exists(os.path.join(fdir, "meta.json")):
            if not prebuild:
                raise RuntimeError(f"data.memmap: the cache {d} has no derived inputs for this run's target factor / "
                                   f"off-shellness ({fdir}); build them with a CPU prebuild job first: {rebuild}")
            nb = meta["N"] * (8 + 4 * len(exp._internal_mass_pdgs))
            _locked_build(fdir, nb, root, lambda st: _build_features(exp, c, d, meta, st, fkd, want_offshell))
    if prebuild:
        LOGGER.info(f"data.memmap_prebuild: the cache {d} is ready{f' (derived inputs {fdir})' if fdir else ''}; "
                    f"exiting before the model")
        sys.exit(0)
    _open_run(exp, c, d, meta, fdir, need, want_offshell)


def _open_run(exp, c, d, meta, fdir, need, want_offshell):
    """Point the experiment at the cache for this run: the selection, the tokens, the statistics (its own, or
    frozen), the per-event arrays in the role's shuffled order, and the batch source."""
    from particle_ids import global_encode, species_pdgs
    from preprocessing import preprocess_amplitude
    from experiment import recipe_process_cfg
    cfg = exp.cfg
    names, roles, specs, paths = c["names"], c["roles"], c["specs"], c["paths"]
    t0 = time.time()
    _check_canary(d, meta, paths)
    if fdir is not None:
        fmeta = json.load(open(os.path.join(fdir, "meta.json")))
        now = _canary(fdir, list(fmeta["canary"]))
        assert now == fmeta["canary"], f"memmap cache {fdir}: derived inputs differ from what the build wrote"
    _touch(d)
    phys_on = bool(cfg.data.get("target_physics_factor", False))
    dimless = bool(cfg.data.get("dimensionless_inputs", False))
    seg_of = {(s["role"], s["name"]): s for s in meta["segs"]}
    pidx = {n: i for i, n in enumerate(names)}
    mom, pdg, amp, rs = (_open(d, n) for n in ("mom", "pdg", "amp", "rs"))
    logF = _open(fdir, "logF") if phys_on else None

    # ---- tokens: the registrations the in-memory path makes as it loads (each pool's distinct ids, ascending, in
    #      load order), then one table per process over every id of the cache ----
    seen = []
    for r in roles:
        for n in names:
            k = need[(r, n)]
            u = [p for p, first in seg_of[(r, n)]["pdg_first"] if first < k]
            if u:
                exp.tokenizer.register_and_encode(np.asarray(u, dtype=int))
                seen += u
    tok_U = np.unique(np.asarray(seen, dtype=np.int64))       # every id a selected event carries
    tok_lut = np.zeros((len(names), len(tok_U)), dtype=np.int64)
    for i, s in enumerate(specs):
        if c["use_pids"]:
            tok_lut[i] = exp.tokenizer.encode(tok_U)
        else:
            tok_lut[i] = global_encode(species_pdgs(tok_U, s.get("physics"), recipe_process_cfg(s)))
    for i, n in enumerate(names):
        exp._proc_npart[i] = int(seg_of[(roles[0], n)]["P"])

    # ---- momentum scale: the train selection's augmented momenta, as the in-memory path reduces them ----
    seed = exp._aug_seed()

    def train_piece(job):
        s, i, j = job
        P = s["P"]
        com = np.asarray(mom[s["p0"] + i * P:s["p0"] + j * P]).reshape(j - i, P, 4)
        a = keyed_aug.augment(com, seed, s["uid"], np.arange(i, j, dtype=np.int64))
        if dimless:
            r_ = np.asarray(rs[s["ev0"] + i:s["ev0"] + j])
            a = (a.astype(np.float64) / r_[:, None, None]).astype(np.float32)
        return a.reshape(-1, 4)
    step_ev = 1 << 16
    jobs = [(seg_of[("train", n)], i, min(need[("train", n)], i + step_ev))
            for n in names for i in range(0, need[("train", n)], step_ev)]

    def train_chunks():         # in order, computed by a few threads ahead (numpy releases the GIL)
        return _ordered_map(train_piece, jobs)
    mom_div, mom_mean, mom_std = c["mom_div"], c["mom_mean"], c["mom_std"]
    if mom_div is None:
        # the run's own scale over its train selection and seed; kept beside the cache (a few numbers, keyed by the
        # selection, the seed and dimensionless_inputs), so a second run on the same data and seed does not redo it
        sk = _hash({"seed": seed, "dimless": dimless, "need": [need[("train", n)] for n in names],
                    "format": MMAP_VERSION})
        sp = os.path.join(d, "mom_stats", sk + ".json")
        if os.path.exists(sp):
            mom_div, mom_mean, mom_std = json.load(open(sp))
        else:
            t1 = time.time()
            _, sd = f32_mean_std(train_chunks)
            mom_div = float(sd)
            mm_, ss_ = f32_mean_std(train_chunks, scale=mom_div)
            mom_mean = float(mm_)
            mom_std = float(max(ss_, 1e-2))
            os.makedirs(os.path.dirname(sp), exist_ok=True)
            tmp = sp + "." + _job_tag()
            with open(tmp, "w") as f:
                json.dump([mom_div, mom_mean, mom_std], f)
            os.replace(tmp, sp)
            LOGGER.info(f"data.memmap: momentum scale over the train selection at seed {seed} in "
                        f"{time.time() - t1:.0f} s")
    exp.mom_mean, exp.mom_std, exp.mom_div = [mom_mean], [mom_std], float(mom_div)

    # ---- dimensionless_inputs: the mass column's statistics over the massive train particles ----
    if dimless:
        dms = c.get("dimless_mass_stats")
        if dms is not None:
            exp._dimless_mass_stats = [float(dms[0]), float(dms[1])]
        else:
            from particle_ids import GLOBAL_PROPERTY_MATRIX, PARTICLE_FEATURE_NAMES, MASS_MASSLESS_LOG10_THR
            raw_lm = np.asarray(GLOBAL_PROPERTY_MATRIX)[:, PARTICLE_FEATURE_NAMES.index("log10_mass_gev")]
            n1 = s1 = s2 = 0.0
            for n in names:
                s = seg_of[("train", n)]
                k, P = need[("train", n)], s["P"]
                pg = np.asarray(pdg[s["p0"]:s["p0"] + k * P])
                t = tok_lut[pidx[n]][np.searchsorted(tok_U, pg)].reshape(k, P)
                r_ = np.asarray(rs[s["ev0"]:s["ev0"] + k])
                v = (raw_lm[t] - np.log10(r_)[:, None])[(raw_lm[t] >= MASS_MASSLESS_LOG10_THR) & (t > 0)]
                n1 += v.size; s1 += float(v.sum()); s2 += float((v.astype(np.float64) ** 2).sum())
            mu = s1 / n1 if n1 else 0.0
            sdv = float(np.sqrt(max(s2 / n1 - mu ** 2, 0.0))) if n1 else 1.0
            exp._dimless_mass_stats = [mu, sdv if sdv > 1e-6 else 1.0]
        LOGGER.info(f"dimensionless_inputs (memmap): log10(m / sqrt(s_hat)) over the massive train particles, mean "
                    f"{exp._dimless_mass_stats[0]:.3f}, std {exp._dimless_mass_stats[1]:.3f}")

    # ---- amplitudes: the target factor, then the shared transform / standardization (frozen or fitted here) ----
    raw, fac = {}, {}
    for r in roles:
        for n in names:
            s, k = seg_of[(r, n)], need[(r, n)]
            a = np.asarray(amp[s["ev0"]:s["ev0"] + k], dtype=np.float64).reshape(-1, 1)
            if phys_on:
                F = np.exp(np.asarray(logF[s["ev0"]:s["ev0"] + k])).reshape(a.shape)
                fac[(r, n)] = F
                a = a * F
            raw[(r, n)] = a
    if phys_on:
        lf = np.concatenate([np.log(fac[("train", n)]).reshape(-1) for n in names])
        LOGGER.info(f"target_physics_factor ON (memmap): {len(names)} processes, mean log F over the train events "
                    f"{lf.mean():.3g}, 5-95% {np.percentile(lf, 5):.3g} to {np.percentile(lf, 95):.3g}")
        del lf
    amp_min = min(float(np.min(v)) for v in raw.values() if v.size)
    per_dataset = c["per_dataset"]
    amp_trafos, prepd_means, prepd_stds = exp._resolve_amp_preprocessing(
        names, roles, lambda r, n: raw.get((r, n)), lambda: np.concatenate([raw[("train", n)] for n in names], axis=0),
        lambda n: seg_of[("train", n)]["P"], per_dataset,
        c["amp_trafos"], c["amp_trafos_pp"], c["prepd_means"], c["prepd_stds"], global_min=amp_min)
    trafos_pp = exp._amp_trafos_pp

    # ---- per-event arrays in each role's shuffled order (rng seed 42 over train, val, test, as in memory) ----
    rng = np.random.default_rng(seed=42)
    ev_rows, amps, pids, facs, counts = [], [], [], [], {}
    exp._role_perm = {}
    for r in roles:
        rr, aa, pp_, ff = [], [], [], []
        for n in names:
            i = pidx[n]
            s, k = seg_of[(r, n)], need[(r, n)]
            m = prepd_means[i] if per_dataset else prepd_means[0]
            sd_ = prepd_stds[i] if per_dataset else prepd_stds[0]
            tr = trafos_pp[i] if trafos_pp else amp_trafos
            ap, _, _ = preprocess_amplitude(raw.pop((r, n)), trafos=tr, mean=m, std=sd_)
            aa.append(ap)
            rr.append(s["ev0"] + np.arange(k, dtype=np.int64))
            pp_.append(np.full(k, i, dtype=np.int32))
            if phys_on:
                ff.append(fac.pop((r, n)).reshape(-1))
        perm = rng.permutation(sum(len(x) for x in rr))
        exp._role_perm[r] = perm
        counts[r] = len(perm)
        ev_rows.append(np.concatenate(rr)[perm])
        amps.append(np.concatenate(aa, axis=0)[perm].astype(np.float32))
        pids.append(np.concatenate(pp_)[perm])
        if phys_on:
            facs.append(np.concatenate(ff)[perm])
    ev_rows = np.concatenate(ev_rows)
    exp.all_amplitudes = np.concatenate(amps, axis=0)
    exp.all_process_ids = np.concatenate(pids)
    exp.all_amp_factor = np.concatenate(facs) if phys_on else None
    del amps, pids, facs

    # ---- order features; the off-shellness columns' statistics over every role (as in memory) ----
    order_rows = np.stack([exp._order_row(i, c["amp_orders"]) for i in range(len(names))]).astype(np.float32)
    F = order_rows.shape[1]
    osh_stats, eps = None, float(cfg.data.get("dimless_offshell_eps", 1e-6))
    src = MmapSource(d, fdir if fdir is not None else d, meta, ev_rows, seed, mom_div, dimless, tok_U, tok_lut,
                     order_rows, None, eps, F)
    if want_offshell:
        src.osh = _open(fdir, "osh")
        n_pdg = src.osh.shape[1]
        step = CHUNK // max(n_pdg, 1)
        mean0, std0 = rows_mean_std(lambda: (src.osh_transformed(ev_rows[i:i + step])
                                             for i in range(0, len(ev_rows), step)))
        mu, sdv = exp._resolve_offshell_stats(lambda: (mean0, std0), n_pdg)
        exp._offshell_stats = [np.asarray(mu).tolist(), np.asarray(sdv).tolist()]
        src.osh_mu, src.osh_sd = np.asarray(mu, np.float64), np.asarray(sdv, np.float64)
        LOGGER.info(f"offshell_per_event ON (memmap): internal-mass slot ({n_pdg} col) filled per batch with "
                    f"signed-log-standardized (s-M^2) for pdgs {exp._internal_mass_pdgs}")
    exp._mmap_src = src
    exp.n_order_features = int(F)
    exp.N_events = int(len(ev_rows))
    exp._role_counts = tuple(int(counts[r]) for r in roles)
    exp._mmap_dir = d
    nbytes = _du(d)
    if bool(cfg.data.get("memmap_prefetch", True)):
        # one sequential read of the arrays: the pages then sit in the page cache (file-backed, shared with the
        # workers, reclaimable), and the shuffled batches do not fault them in one random page at a time
        t1 = time.time()
        for a in ("mom", "pdg", "rs", "evp"):
            with open(os.path.join(d, a + ".npy"), "rb", buffering=0) as fh:
                while fh.read(64 << 20):
                    pass
        LOGGER.info(f"data.memmap: prefetched the batch arrays into the page cache in {time.time() - t1:.0f} s")
    LOGGER.info(f"data.memmap: {exp.N_events} events of {d} ({nbytes / 1e9:.2f} GB on disk), roles {exp._role_counts}; "
                f"opened in {time.time() - t0:.0f} s (augmentation seed {seed}, momentum scale {mom_div:.6g})")
