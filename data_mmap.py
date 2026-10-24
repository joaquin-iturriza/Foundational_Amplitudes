"""data.memmap: the recipe data path with host memory that scales with the batch, not the dataset.

The in-memory path (experiment._init_data_recipes) holds every pool's boosted momenta, their concatenations and the
per-event arrays in RAM at once, with several transient copies (tools/mem_profile_data.py measures them). Here each
(role, process) pool is loaded, boosted and written once, in the same order and with the same draws of the torch RNG
as the in-memory path, into flat arrays on disk; the statistics are reduced in chunks with the exact summation order
numpy uses on the concatenated arrays (so they are bitwise the in-memory path's); the shuffled per-event arrays are
written in their final order. The run then opens the arrays with np.load(mmap_mode="r") and every loader (and every
dataloader worker, which shares the file pages) gathers its batch by index (MmapAmplitudeDataset).

The cache lives under $SCRATCH/amp_mmap_cache/<key> (data.memmap_dir overrides the root). The key is a hash of
everything the arrays depend on: the pools (path, size, mtime, recipe_id), the per-role caps, the recipe's per-process
order rows and propagator masks, the preprocessing flags, the frozen statistics a fine-tune / resume / rebuild loads
(or "fit here"), the tokenizer's starting vocabulary and the torch RNG state at entry (which carries the run's seed and
anything drawn before). The full key document is stored with the arrays and compared on every open, so a cache built
from other inputs is never read. A cache hit restores the torch RNG state the build left, so the model initialisation
draws what it would have drawn after the in-memory path.

Not on this path (asserted): data.target_propagators, training.sign_head, data.resonance_per_event,
data.phase_space_on_pool. data.target_physics_factor is: the factor multiplies |M|^2 as each pool loads and is kept
per event in amp_factor.npy, in the final order, so the evaluation divides it out as on the in-memory path.
"""
import fcntl
import hashlib
import json
import os
import shutil
import time

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

import logging.handlers  # noqa: F401  (logger.py uses logging.handlers without importing it)

from logger import LOGGER

MMAP_VERSION = 1
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
class MmapAmplitudeDataset(Dataset):
    """AmplitudeDataset over memory-mapped per-event / per-particle arrays, restricted to `rows` (a range, never
    materialised, or an int64 array of rows). A batch is gathered from the maps (numpy reads only the pages it
    touches; the pages are the file's, shared with every dataloader worker) and returned exactly as
    AmplitudeDataset.__getitems__ returns it, the integer arrays widened to int64 and the floats cast to `dtype`."""

    def __init__(self, particles_flat, offsets, amplitudes, tokens_flat, order_labels, dtype, process_ids, rows):
        self.particles_flat, self.offsets, self.amplitudes = particles_flat, offsets, amplitudes
        self.tokens_flat, self.order_labels, self.process_ids = tokens_flat, order_labels, process_ids
        self.dtype = dtype
        self.rows = rows if isinstance(rows, range) else np.asarray(rows, dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def _rows(self, idx):
        if isinstance(self.rows, range):
            assert self.rows.step == 1
            return idx + self.rows.start
        return self.rows[idx]

    def _f(self, a):
        a = np.ascontiguousarray(a)
        return torch.from_numpy(a if a.flags.writeable else a.copy()).to(self.dtype)

    def __getitems__(self, indices):
        from dataset import _Prebatched
        if os.environ.get("LLOCA_FETCH", "batch") == "item":
            return [self[int(i)] for i in indices]
        idx = np.asarray(indices, dtype=np.int64)
        r = self._rows(idx)
        off = np.asarray(self.offsets[r])
        starts = off[:, 0].astype(np.int64)
        counts = (off[:, 1] - off[:, 0]).astype(np.int64)
        csum = np.cumsum(counts)
        total = int(csum[-1]) if len(csum) else 0
        gather = np.repeat(starts - (csum - counts), counts) + np.arange(total, dtype=np.int64)
        ptr = torch.zeros(len(idx) + 1, dtype=torch.long)
        ptr[1:] = torch.from_numpy(csum)
        return [_Prebatched((
            self._f(self.particles_flat[gather]),
            self._f(self.amplitudes[r]),
            torch.from_numpy(np.asarray(self.tokens_flat[gather]).astype(np.int64)),
            self._f(self.order_labels[r]),
            ptr,
            torch.from_numpy(np.asarray(self.process_ids[r]).astype(np.int64)),
        ))]

    def __getitem__(self, i):
        r = int(self._rows(np.int64(i)))
        start, end = int(self.offsets[r, 0]), int(self.offsets[r, 1])
        return (
            self._f(self.particles_flat[start:end]),
            self._f(self.amplitudes[r]),
            torch.from_numpy(np.asarray(self.tokens_flat[start:end]).astype(np.int64)),
            self._f(self.order_labels[r]),
            torch.tensor(int(self.process_ids[r]), dtype=torch.long),
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
def _pool_doc(path):
    st = os.stat(path)
    rid = None
    for side in (path + ".recipe.json", os.path.splitext(path)[0] + ".recipe.json"):
        if os.path.exists(side):
            try:
                rid = json.load(open(side)).get("recipe_id")
            except (OSError, ValueError):
                pass
            break
    return [os.path.abspath(path), int(st.st_size), int(st.st_mtime_ns), rid]


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


def _code_hash():
    """Hash of the code the cached arrays come from: this module, the preprocessing and particle tables, and the
    experiment methods that load, boost, order and standardize (a change to any of them builds a new cache)."""
    import inspect
    import experiment
    import multiplicity_extrap
    import particle_ids
    import preprocessing
    from lloca.utils import polar_decomposition, rand_transforms
    E = experiment.AmplitudeExperiment
    import physics_scales
    parts = [inspect.getsource(m) for m in (
        __import__(__name__), preprocessing, particle_ids, multiplicity_extrap, rand_transforms, polar_decomposition,
        physics_scales)]
    parts += [inspect.getsource(f) for f in (
        E._boost_augment_momenta, E._order_row, E._resolve_amp_preprocessing, E._resolve_offshell_stats,
        E._setup_offshell_masks, E._physics_logF, E._eikonal_logS)]
    parts += [np.__version__, torch.__version__]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _file_digest(exp, path):
    """sha256 of a file the cached targets depend on (the coupling table of data.target_physics_factor), resolved as
    experiment._physics_logF resolves it"""
    p = os.path.expandvars(str(path))
    p = p if os.path.isabs(p) else os.path.join(os.path.dirname(os.path.abspath(experiment_file())), p)
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def experiment_file():
    import experiment
    return experiment.__file__


def _cache_root(cfg):
    root = cfg.data.get("memmap_dir", None)
    if root in (None, "null", "none", ""):
        root = os.path.join(os.environ.get("SCRATCH") or os.path.join(cfg.run_dir, ".."), "amp_mmap_cache")
    return os.path.abspath(str(root))


def _open(d, name):
    return np.load(os.path.join(d, name + ".npy"), mmap_mode="r")


def init_data_recipes_mmap(exp, c):
    """The memory-mapped twin of the tail of AmplitudeExperiment._init_data_recipes (from the per-role loads on).
    `c` holds the locals that function computed before: specs, names, roles, paths, use_pids, per_dataset, the
    frozen statistics (None when fitted here), amp_orders, n_order_features, cap_for, stats_path, stats_on_pool."""
    from particle_ids import global_encode
    from preprocessing import resolve_amp_trafos, preprocess_amplitude
    cfg = exp.cfg
    names, roles, specs = c["names"], c["roles"], c["specs"]

    assert not c["stats_on_pool"], "data.memmap: data.phase_space_on_pool is not implemented on the memmap path"
    assert not bool(cfg.data.get("dimensionless_inputs", False)), \
        "data.memmap: data.dimensionless_inputs is implemented on the in-memory path only (data.memmap=false)"
    assert not bool(cfg.data.get("target_physics_orders", False)), \
        "data.memmap: data.target_physics_orders (per-event alpha_s) is implemented on the in-memory path only"
    for k in ("target_propagators", "resonance_per_event"):
        assert not bool(cfg.data.get(k, False)), f"data.memmap: data.{k} is not implemented on the memmap path"
    assert not bool(cfg.training.get("sign_head", False)), "data.memmap: training.sign_head is not implemented"
    phys_on = bool(cfg.data.get("target_physics_factor", False))
    exp._target_factor_on = phys_on
    exp._sign_head = False

    _porder = str(cfg.data.get("particle_order", "none"))
    want_offshell = (exp._offshell_per_event and exp._use_internal_mass_scalars and bool(exp._internal_mass_pdgs))
    # the propagator masks need the pools' slot PDGs (set while loading), so _build makes them; the key holds
    # what they are made from: the diagram sidecars, the property-table flags and the scanned masses
    ddir = cfg.model.get("diagrams_dir", "data/diagrams")
    if not os.path.isabs(ddir):
        ddir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ddir)
    sidecars = [exp._resolve_diagram_path(ddir, n) for n in names] if want_offshell else []
    order_rows = np.stack([exp._order_row(i, c["amp_orders"]) for i in range(len(names))]).astype(np.float32)
    ft = cfg.get("fine_tune", None)
    ft_on = ft is not None and ft.get("pretrained_path", None) is not None and not exp.warm_start
    key_doc = {
        "version": MMAP_VERSION,
        "code": _code_hash(),
        "names": names,
        "pools": {r: [_pool_doc(c["paths"][r][n]) for n in names] for r in roles},
        "caps": {r: c["cap_for"][r] for r in roles},
        "particle_order": _porder,
        "use_pids": bool(c["use_pids"]),
        "tokenizer0": sorted((int(k), int(v)) for k, v in exp.tokenizer._pdg_to_idx.items()),
        "rng_in": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "frozen": _jsonable({k: c[k] for k in ("mom_div", "mom_mean", "mom_std", "amp_trafos", "amp_trafos_pp",
                                                "prepd_means", "prepd_stds")}),
        "per_dataset": bool(c["per_dataset"]),
        "base_trafos": list(cfg.data.amp_trafos),
        "signedlog_quantile": exp._signedlog_quantile(),
        "signedlog_scale_shift": exp._signedlog_scale_shift(),
        "shared_standardization": bool(cfg.data.get("shared_standardization", False)),
        "affine_multiplicity_std": bool(cfg.data.get("affine_multiplicity_std", False)),
        "target_physics_factor": ([float(cfg.data.get("target_physics_s0", 289.0)),
                                   str(cfg.data.get("target_physics_couplings", "default")),
                                   _file_digest(exp, cfg.data.get("target_physics_couplings",
                                                                  "analysis/catalog_daniel/data/diagram_couplings.json")),
                                   bool(cfg.data.get("target_physics_dim_only", False)),
                                   str(cfg.data.get("target_physics_dim_ref", "shat")),
                                   bool(cfg.data.get("target_physics_eikonal", False)),
                                   float(cfg.data.get("target_physics_qperp_tau", 0.3))] if phys_on else None),
        "order_rows": _jsonable(order_rows),
        "offshell": _jsonable({
            "on": want_offshell, "pdgs": exp._internal_mass_pdgs,
            "sidecars": [None if p is None else _pool_doc(p)[:3] for p in sidecars],
            "table": {k: bool(cfg.data.get(k, dflt)) for k, dflt in (
                ("spin_onehot", False), ("color_onehot", False), ("prop_is_massless", True),
                ("standardize_props", True), ("generation_onehot", False), ("generation_feature", True))},
            "k_pe": int((cfg.model.get("diagram_encoder", {}) or {}).get("k_pe", 8)),
            "internal_masses": getattr(exp, "_internal_mass_by_proc", None),
            "pinned": (None if cfg.data.get("offshell_stats", None) is None
                       else np.asarray(cfg.data.offshell_stats, dtype=np.float64).tolist()),
            "ft_mode": str(ft.get("offshell_stats", "parent")) if ft_on else None,
            "parent": getattr(exp, "_parent_offshell", None) if ft_on else None,
            "ft_parent_cfg": (None if not ft_on or ft.get("parent_offshell_stats", None) is None
                              else np.asarray(ft.parent_offshell_stats, dtype=np.float64).tolist())}),
    }
    key = hashlib.sha256(json.dumps(key_doc, sort_keys=True).encode()).hexdigest()[:24]
    root = _cache_root(cfg)
    d = os.path.join(root, key)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, key + ".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)              # one builder per key; the others wait and then read it
        meta_p = os.path.join(d, "meta.json")
        built = not os.path.exists(meta_p)
        if not built:
            meta = json.load(open(meta_p))
            assert meta["key_doc"] == json.loads(json.dumps(key_doc, sort_keys=True)), \
                f"memmap cache {d}: stored key document differs from this run's (hash collision?)"
            LOGGER.info(f"data.memmap: opening the cache {d}")
        else:
            t0 = time.time()
            tmp = d + f".tmp{os.getpid()}"
            # a builder that died left its staging dir; under the lock no other builder of this key is alive
            for old in os.listdir(root):
                if old.startswith(key + ".tmp"):
                    shutil.rmtree(os.path.join(root, old), ignore_errors=True)
            os.makedirs(tmp)
            meta = _build(exp, c, tmp, key_doc, order_rows, want_offshell,
                          global_encode, resolve_amp_trafos, preprocess_amplitude)
            with open(os.path.join(tmp, "meta.json"), "w") as f:
                json.dump(meta, f)
            os.rename(tmp, d)
            LOGGER.info(f"data.memmap: built the cache {d} in {time.time() - t0:.0f} s")
    _apply(exp, c, d, meta, built)


def _build(exp, c, d, key_doc, order_rows, want_offshell, global_encode, resolve_amp_trafos,
           preprocess_amplitude):
    from experiment import order_identical_particles
    from particle_ids import species_pdgs
    cfg = exp.cfg
    names, roles, specs, paths = c["names"], c["roles"], c["specs"], c["paths"]
    use_pids = c["use_pids"]
    _porder = key_doc["particle_order"]
    _porder_rng = np.random.default_rng(int(cfg.data.get("seed", 42))) if _porder == "shuffle" else None

    # ---- pass 1: shapes (row counts from the .npy headers), so the flat files can be allocated up front ----
    shape = {}
    for role in roles:
        cap = c["cap_for"][role]
        cap = int(cap) if cap not in (None, "null", "none") else None
        for n in names:
            N0, ncol = np.load(paths[role][n], mmap_mode="r").shape
            shape[(role, n)] = (min(N0, cap) if cap is not None else N0, (ncol - 1) // 5)
    n_role = {r: sum(shape[(r, n)][0] for n in names) for r in roles}
    m_role = {r: sum(shape[(r, n)][0] * shape[(r, n)][1] for n in names) for r in roles}
    M, N = sum(m_role.values()), sum(n_role.values())
    fmt = np.lib.format
    particles = fmt.open_memmap(os.path.join(d, "particles.npy"), "w+", np.float32, (M, 4))
    tokens = fmt.open_memmap(os.path.join(d, "tokens.npy"), "w+", np.int16, (M,))

    # ---- pass 2: load, tokenize, boost + augment (the torch RNG draws in the in-memory order), write ----
    raw_amp = {r: fmt.open_memmap(os.path.join(d, f"stage_amp_{r}.npy"), "w+", np.float64, (n_role[r], 1))
               for r in roles}
    phys_on = bool(exp.cfg.data.get("target_physics_factor", False))
    fac = ({r: fmt.open_memmap(os.path.join(d, f"stage_factor_{r}.npy"), "w+", np.float64, (n_role[r], 1))
            for r in roles} if phys_on else None)
    reg, slot_pdgs, npart = [], {}, {}
    pbase = ebase = 0
    seg = {}                                         # (role, name) -> (particle offset, event offset in the role, N, P)
    for role in roles:
        cap = c["cap_for"][role]
        cap = int(cap) if cap not in (None, "null", "none") else None
        er = 0
        for s in specs:
            name = s["name"]
            if cap is not None:
                data_raw = np.asarray(np.load(paths[role][name], mmap_mode="r")[:cap])
            else:
                data_raw = np.load(paths[role][name])
            n_particles = (data_raw.shape[1] - 1) // 5
            momenta_cols = n_particles * 4
            parts = data_raw[:, :momenta_cols]
            pdg_ids = data_raw[:, momenta_cols:-1].astype(int)
            amplitudes = data_raw[:, [-1]]
            if use_pids:
                toks = exp.tokenizer.register_and_encode(pdg_ids)
            else:
                toks = global_encode(species_pdgs(pdg_ids, s.get("physics")))   # heavy-quark tops: their own mass
                exp.tokenizer.register_and_encode(pdg_ids)
            reg.append(np.unique(pdg_ids).tolist())
            slot_pdgs.setdefault(name, [int(x) for x in pdg_ids[0]])
            if _porder != "none":
                parts = order_identical_particles(parts, pdg_ids, _porder, rng=_porder_rng)
            mom = exp._boost_augment_momenta(parts, n_particles)
            Nn, P = mom.shape[0], mom.shape[1]
            assert (Nn, P) == shape[(role, name)]
            toks = np.asarray(toks).reshape(Nn * P)
            assert toks.min() >= np.iinfo(np.int16).min and toks.max() <= np.iinfo(np.int16).max
            particles[pbase:pbase + Nn * P] = mom.reshape(Nn * P, 4)
            tokens[pbase:pbase + Nn * P] = toks
            if phys_on:      # data.target_physics_factor: s_hat is boost invariant, so the lab momenta serve
                F_ev = np.exp(exp._physics_logF(name, parts.reshape(len(parts), n_particles, 4))).reshape(-1, 1)
                amplitudes = amplitudes * F_ev
                fac[role][er:er + Nn] = F_ev
            raw_amp[role][er:er + Nn] = amplitudes
            seg[(role, name)] = (pbase, er, Nn, P)
            npart[name] = int(P)
            pbase += Nn * P; er += Nn
            LOGGER.info(f"  [{role}] {name}: {Nn} events{f' (cap {cap})' if cap else ''} from {paths[role][name]}"
                        f" -> memmap")
            del data_raw, parts, mom, toks
    exp._slot_pdgs_by_name = {**slot_pdgs, **getattr(exp, "_slot_pdgs_by_name", {})}
    offshell_masks = exp._setup_offshell_masks(names) if want_offshell else None
    rng_out = torch.get_rng_state().numpy().copy()
    np.save(os.path.join(d, "rng_out.npy"), rng_out)
    tokens.flush(); del tokens

    # ---- momentum scale: one global train-only std (bitwise the in-memory np.concatenate(...).std()) ----
    mom_div, mom_mean, mom_std = c["mom_div"], c["mom_mean"], c["mom_std"]
    m_tr = m_role["train"]

    def train_chunks():
        step = CHUNK // 4
        for i in range(0, m_tr, step):
            yield np.asarray(particles[i:min(m_tr, i + step)])
    if mom_div is None:
        _, sd = f32_mean_std(train_chunks)
        mom_div = float(sd)
        mm_, ss_ = f32_mean_std(train_chunks, scale=mom_div)
        mom_mean = float(mm_)
        mom_std = float(max(ss_, 1e-2))
    for i in range(0, M, CHUNK // 4):                              # every block / mom_div, in place
        j = min(M, i + CHUNK // 4)
        particles[i:j] = np.asarray(particles[i:j]) / mom_div
    particles.flush()

    # ---- amplitude transform and statistics (per process; the shared helper of the in-memory path) ----
    def get_amp(role, name):
        _, er, Nn, _ = seg[(role, name)]
        return np.asarray(raw_amp[role][er:er + Nn])
    amp_min = min(float(np.min(raw_amp[r])) for r in roles)
    per_dataset = c["per_dataset"]
    amp_trafos, prepd_means, prepd_stds = exp._resolve_amp_preprocessing(
        names, roles, get_amp, lambda: np.asarray(raw_amp["train"]), lambda n: npart[n], per_dataset,
        c["amp_trafos"], c["amp_trafos_pp"], c["prepd_means"], c["prepd_stds"], global_min=amp_min)
    trafos_pp = exp._amp_trafos_pp
    affine = getattr(exp, "_affine_multiplicity", None)

    # ---- per-event arrays in the role's shuffled order (rng seed 42 over train, val, test, as in memory) ----
    F = order_rows.shape[1]
    offsets = fmt.open_memmap(os.path.join(d, "offsets.npy"), "w+", np.int64, (N, 2))
    amp_out = fmt.open_memmap(os.path.join(d, "amplitudes.npy"), "w+", np.float32, (N, 1))
    fac_out = fmt.open_memmap(os.path.join(d, "amp_factor.npy"), "w+", np.float64, (N, 1)) if phys_on else None
    pid_out = fmt.open_memmap(os.path.join(d, "process_ids.npy"), "w+", np.int32, (N,))
    order_out = fmt.open_memmap(os.path.join(d, "order_labels.npy"), "w+", np.float32, (N, F))
    n_pdg = len(exp._internal_mass_pdgs)
    osh_raw = (fmt.open_memmap(os.path.join(d, "stage_offshell.npy"), "w+", np.float32, (N, n_pdg))
               if want_offshell else None)
    rng = np.random.default_rng(seed=42)
    part_base = 0
    row0 = 0
    for role in roles:
        nr = n_role[role]
        st_amp = fmt.open_memmap(os.path.join(d, f"stage_prepd_{role}.npy"), "w+", np.float64, (nr, 1))
        st_osh = (fmt.open_memmap(os.path.join(d, f"stage_osh_{role}.npy"), "w+", np.float32, (nr, n_pdg))
                  if want_offshell else None)
        ev_start = np.zeros(len(names) + 1, dtype=np.int64)     # first event of each process in the role
        p_base = np.zeros(len(names), dtype=np.int64)           # its first particle in the flat array
        p_np = np.zeros(len(names), dtype=np.int64)
        for proc_idx, name in enumerate(names):
            pb, er, Nn, P = seg[(role, name)]
            exp._proc_npart[proc_idx] = int(P)
            m = prepd_means[proc_idx] if per_dataset else prepd_means[0]
            s_ = prepd_stds[proc_idx] if per_dataset else prepd_stds[0]
            tr = trafos_pp[proc_idx] if trafos_pp else amp_trafos
            ap, _, _ = preprocess_amplitude(get_amp(role, name), trafos=tr, mean=m, std=s_)
            st_amp[er:er + Nn] = ap
            ev_start[proc_idx + 1] = ev_start[proc_idx] + Nn
            assert pb == part_base
            p_base[proc_idx] = pb
            p_np[proc_idx] = P
            if want_offshell:
                cols = np.zeros((Nn, n_pdg), dtype=np.float64)
                sel = offshell_masks[proc_idx] if offshell_masks is not None else None
                if sel is not None:
                    pp = np.asarray(particles[pb:pb + Nn * P]).reshape(Nn, P, 4).astype(np.float64)
                    for col, pdg in enumerate(exp._internal_mass_pdgs):
                        props = sel.get(pdg, [])
                        if not props:
                            continue
                        offs = []
                        for mrow, mm in props:
                            q = np.einsum("s,nsc->nc", mrow, pp) * mom_div
                            s_prop = q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(axis=1)
                            offs.append(s_prop - mm)
                        offs = np.stack(offs, axis=1)
                        idx = np.argmin(np.abs(offs), axis=1)
                        cols[:, col] = offs[np.arange(Nn), idx]
                st_osh[er:er + Nn] = cols.astype(np.float32)
            part_base += Nn * P
        perm = rng.permutation(nr)
        np.save(os.path.join(d, f"perm_{role}.npy"), perm)
        for i in range(0, nr, CHUNK):
            src = perm[i:i + CHUNK]
            p = np.searchsorted(ev_start, src, side="right") - 1
            start = p_base[p] + (src - ev_start[p]) * p_np[p]
            o = slice(row0 + i, row0 + i + len(src))
            offsets[o] = np.stack([start, start + p_np[p]], axis=1)
            amp_out[o] = np.asarray(st_amp[src]).astype(np.float32)
            if phys_on:
                fac_out[o] = np.asarray(fac[role][src])
            pid_out[o] = p.astype(np.int32)
            order_out[o] = order_rows[p]
            if want_offshell:
                osh_raw[o] = st_osh[src]
        row0 += nr
        del st_amp, st_osh, perm
        for f in (f"stage_prepd_{role}.npy", f"stage_osh_{role}.npy"):
            if os.path.exists(os.path.join(d, f)):
                os.remove(os.path.join(d, f))
    del raw_amp
    for r in roles:
        os.remove(os.path.join(d, f"stage_amp_{r}.npy"))
    if phys_on:
        fac_out.flush()
        lf = np.log(np.asarray(fac["train"]).reshape(-1))
        LOGGER.info(f"target_physics_factor ON (memmap): {len(names)} processes, mean log F over the train events "
                    f"{lf.mean():.3g}, 5-95% {np.percentile(lf, 5):.3g} to {np.percentile(lf, 95):.3g}")
        del fac
        for r in roles:
            os.remove(os.path.join(d, f"stage_factor_{r}.npy"))

    offshell_stats = None
    if want_offshell:
        def osh_chunks():
            for i in range(0, N, CHUNK // max(n_pdg, 1)):
                o = np.asarray(osh_raw[i:i + CHUNK // max(n_pdg, 1)]).astype(np.float64)
                yield np.sign(o) * np.log1p(np.abs(o))
        mean0, std0 = rows_mean_std(osh_chunks)
        mu, sd = exp._resolve_offshell_stats(lambda: (mean0, std0), n_pdg)
        offshell_stats = [mu.tolist(), sd.tolist()]
        i = 0
        for t in osh_chunks():
            order_out[i:i + len(t), -n_pdg:] = ((t - mu) / sd).astype(np.float32)
            i += len(t)
        del osh_raw
        os.remove(os.path.join(d, "stage_offshell.npy"))
    for a in (particles, offsets, amp_out, pid_out, order_out):
        a.flush()
    return {
        "key_doc": json.loads(json.dumps(key_doc, sort_keys=True)),
        "n_role": n_role, "m_role": m_role, "n_order_features": int(F),
        "register": reg, "slot_pdgs": slot_pdgs, "npart": [npart[n] for n in names],
        "mom_div": mom_div, "mom_mean": mom_mean, "mom_std": mom_std,
        "amp_trafos": amp_trafos, "amp_trafos_pp": trafos_pp, "per_dataset": per_dataset,
        "prepd_mean": [float(x) for x in prepd_means], "prepd_std": [float(x) for x in prepd_stds],
        "affine_multiplicity": affine, "offshell_stats": offshell_stats,
    }


def _apply(exp, c, d, meta, built):
    """Point the experiment at the cache: the arrays as read-only maps, the statistics and the side state
    (tokenizer, slot PDGs, particle counts, the RNG state the build left) exactly as the in-memory path sets them."""
    cfg = exp.cfg
    names, roles = c["names"], c["roles"]
    if not built:                    # the build registered them as it loaded; a hit replays the same calls
        for pdgs in meta["register"]:
            exp.tokenizer.register_and_encode(np.asarray(pdgs, dtype=int))
    if not hasattr(exp, "_slot_pdgs_by_name"):
        exp._slot_pdgs_by_name = {}
    for n in names:
        exp._slot_pdgs_by_name.setdefault(n, meta["slot_pdgs"][n])
    for i, P in enumerate(meta["npart"]):
        exp._proc_npart[i] = int(P)
    torch.set_rng_state(torch.from_numpy(np.load(os.path.join(d, "rng_out.npy"))))

    exp.mom_mean, exp.mom_std, exp.mom_div = [meta["mom_mean"]], [meta["mom_std"]], float(meta["mom_div"])
    exp._amp_trafos_pp = [list(t) for t in meta["amp_trafos_pp"]] if meta["amp_trafos_pp"] else None
    cfg.data.amp_trafos = meta["amp_trafos"]
    exp.prepd_mean, exp.prepd_std = meta["prepd_mean"], meta["prepd_std"]
    if meta.get("affine_multiplicity"):
        exp._affine_multiplicity = meta["affine_multiplicity"]
    if meta.get("offshell_stats") is not None:
        exp._offshell_stats = meta["offshell_stats"]
    if not built:
        LOGGER.info(
            f"Preprocessing amplitudes "
            f"{'per-dataset (independent unit scale)' if meta['per_dataset'] else 'globally (shared scale)'} "
            f"using trafos={meta['amp_trafos']} (train-only stats); "
            f"means={[f'{m:.3f}' for m in meta['prepd_mean']]}")

    exp.particles_flat = _open(d, "particles")
    exp.tokens_flat = _open(d, "tokens")
    exp.offsets = _open(d, "offsets")
    exp.all_amplitudes = _open(d, "amplitudes")
    exp.all_process_ids = _open(d, "process_ids")
    exp.all_order_labels = _open(d, "order_labels")
    exp.all_amp_factor = (np.asarray(_open(d, "amp_factor"))
                          if os.path.exists(os.path.join(d, "amp_factor.npy")) else None)
    assert (exp.all_amp_factor is not None) == bool(cfg.data.get("target_physics_factor", False)), \
        "data.memmap: the cache's amp_factor does not match data.target_physics_factor"
    exp._role_perm = {r: np.load(os.path.join(d, f"perm_{r}.npy"), mmap_mode="r") for r in roles}
    exp.n_order_features = int(meta["n_order_features"])
    exp.N_events = int(exp.offsets.shape[0])
    exp._role_counts = tuple(int(meta["n_role"][r]) for r in roles)
    exp._mmap_dir = d
    nbytes = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
    if bool(cfg.data.get("memmap_prefetch", True)):
        # one sequential read of every array: the pages then sit in the page cache (file-backed, shared with the
        # workers, reclaimable), and the shuffled batches do not fault them in one random page at a time
        t0 = time.time()
        for f in sorted(os.listdir(d)):
            with open(os.path.join(d, f), "rb", buffering=0) as fh:
                while fh.read(64 << 20):
                    pass
        LOGGER.info(f"data.memmap: prefetched {nbytes / 1e9:.2f} GB into the page cache in {time.time() - t0:.0f} s")
    LOGGER.info(f"data.memmap: {exp.N_events} events, {exp.particles_flat.shape[0]} particles mapped from {d} "
                f"({nbytes / 1e9:.2f} GB on disk); roles {exp._role_counts}")
