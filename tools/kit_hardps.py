"""The KIT hard phase-space benchmark (hardps_benchmark, Bakar et al.) read as FA recipe pools, and its scoring layout.

The benchmark ships one HDF5 file per process, `<file>.h5` under $KIT_HARDPS_DIR (default
$DATA_DIR/kit_hardps/hardps_benchmark/data/full): groups `<split>/<stratum>` (bulk_train, bulk_test, hard_train,
hard_test), each holding `momenta` (N, n, 4) float64 (E, px, py, pz) in GeV, CM frame, order (e+, e-, finals), and per
target `<target>` with `log10_<target>` or `log10_abs_<target>`; the root attribute `metadata` (JSON) lists the
particles (pdg, Q, spin, mass, ...) and the targets with their loop count and coupling powers. Nothing here assumes
the three files of the first release: a new file (two-loop gg->HH, hadronic tt, ee->4 jets) with signed targets is
read the same way.

A dataset name is `kit_<file>__<target>[__hard]`, e.g. `kit_ee_tt__virt_finite`, `kit_ee_uug__me2`:
  train  the first n_train events of bulk_train (with `__hard`: plus the first 90% of every hard_train stratum,
         protocol B, where hard points are allowed)
  val    the last n_val events of bulk_train (bulk-only early stopping); with `__hard` plus the last 10% of every
         hard_train stratum, so protocol B's checkpoint sees the hard fit it tests
  test   the first n_test events of bulk_test
Rows are FA's pool rows, float64: the 4-momenta reordered to the catalog's (e-, e+, finals), the PDG ids, the target.
A target with n_loops > 0 is stored alpha_s-stripped as the catalog's one-loop pools are (mg5_pipeline_final
.order_vector): value / (alpha_s / 2 pi)^n_loops, so its order vector is [n_loops, 0, alpha_s power - n_loops, alpha
power] (ee->tt virtual: [1, 0, 0, 2], the catalog's ee_ttbar_nlo). Signed targets are stored signed (the per-dataset
preprocessing takes the signed log).
"""
import json
import os
import sys

import numpy as np

PREFIX = "kit_"
HARD = "hard"
HARD_VAL_FRAC = 0.1
SPLITS = ("bulk_train", "bulk_test", "hard_train", "hard_test")
# the catalog sidecar (data/diagrams/<name>.diagrams.json) each KIT (file, target) reads its diagrams and its
# propagator masks (data.offshell_per_event) from: the same process at the same order in the catalog's leg order. A
# file not listed has none (a run of it sets model.use_diagrams=false and data.offshell_per_event=false, or a sidecar
# is dumped with tools/dump_diagrams.py and listed here).
DIAGRAMS_OF = {("ee_tt", "born"): "ee_ttbar", ("ee_tt", "virt_finite"): "ee_ttbar_nlo",
               ("ee_uug", "me2"): "ee_uug", ("ee_uugg", "me2"): "ee_uugg"}


def kit_dir():
    d = os.environ.get("KIT_HARDPS_DIR")
    if d:
        return d
    return os.path.join(os.environ["DATA_DIR"], "kit_hardps", "hardps_benchmark", "data", "full")


def is_kit(name):
    return str(name).startswith(PREFIX) and "__" in str(name)


def parse(name):
    """(file stem, target, variant) of a dataset name; variant is None or 'hard'."""
    assert is_kit(name), f"{name}: not a KIT dataset name (kit_<file>__<target>[__hard])"
    parts = str(name)[len(PREFIX):].split("__")
    if len(parts) == 3 and parts[2] == HARD:
        return parts[0], parts[1], HARD
    assert len(parts) == 2, f"{name}: kit_<file>__<target>[__hard]"
    return parts[0], parts[1], None


def h5_path(stem):
    return os.path.join(kit_dir(), f"{stem}.h5")


def metadata(stem):
    import h5py
    with h5py.File(h5_path(stem), "r") as f:
        return json.loads(f.attrs["metadata"])


def beam_perm(pdgs):
    """The leg order putting the e- first, as the catalog's e+e- entries (pdg_ids [11, -11, ...]); identity otherwise."""
    p = list(range(len(pdgs)))
    if len(pdgs) >= 2 and int(pdgs[0]) == -11 and int(pdgs[1]) == 11:
        p[0], p[1] = 1, 0
    return p


def target_info(meta, target):
    """(n_loops, alpha_s power, alpha power, alpha_s) of `target` from the file's metadata."""
    t = meta["targets"][target]
    return int(t.get("n_loops", 0)), int(t["alpha_s_power"]), int(t["alpha_power"]), float(meta["alpha_s"])


def strip_factor(meta, target):
    """What the stored target is divided by: (alpha_s / 2 pi)^n_loops (1 at tree level)."""
    n, _, _, a_s = target_info(meta, target)
    return (a_s / (2 * np.pi)) ** n


def catalog_entry(name):
    """The mg5_pipeline_final.PROCESSES entry of a KIT dataset (kind 'kit'): what the recipe path reads of a process."""
    stem, target, variant = parse(name)
    meta = metadata(stem)
    parts = meta["particles"]
    perm = beam_perm([p["pdg"] for p in parts])
    pdgs = [int(parts[i]["pdg"]) for i in perm]
    n_loops, k, b, _ = target_info(meta, target)
    entry = {
        "kind": "kit",
        "kit": {"file": stem, "target": target, "variant": variant},
        "nfinal": len(pdgs) - 2,
        "n_loops": n_loops,
        "alphas_power": k,
        "order": [n_loops, 0, k - n_loops, b],
        "pdg_ids": pdgs,
        "m_finals": [float(parts[i]["mass"]) for i in perm[2:]],
        "param_card_patches": {},
    }
    if (stem, target) in DIAGRAMS_OF:
        entry["diagrams_of"] = DIAGRAMS_OF[(stem, target)]
    return entry


def file_sha256(path):
    """sha256 of a benchmark file, cached beside it in <file>.sha256 keyed on its size and mtime (hashed once per site
    and release: a corrected release of the same size gets a new identity)"""
    import hashlib
    st = os.stat(path)
    key = f"{st.st_size} {st.st_mtime_ns}"
    side = path + ".sha256"
    try:
        k, h = open(side).read().split("\n")[:2]
        if k == key:
            return h
    except (OSError, ValueError):
        pass
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            d.update(b)
    h = d.hexdigest()
    try:
        tmp = f"{side}.tmp{os.getpid()}"
        open(tmp, "w").write(f"{key}\n{h}\n")
        os.replace(tmp, side)
    except OSError:
        pass
    return h


def recipe(name, sqrts_min, sqrts_max, n_events, role, seed):
    """The identity of a KIT pool (recipe_id hashes it): the file's sha256, the target, the variant, the role and the
    events."""
    stem, target, variant = parse(name)
    cfg = catalog_entry(name)
    return {
        "process": name,
        "mode": "kit",
        "sqrts_min": float(sqrts_min),
        "sqrts_max": float(sqrts_max),
        "n_events": int(n_events),
        "role": role,
        "seed": seed,
        "pdg_ids": cfg["pdg_ids"],
        "m_finals": cfg["m_finals"],
        "alphas_power": cfg["alphas_power"],
        "amp_orders": cfg["order"],
        "kit_file": f"{stem}.h5",
        "kit_file_sha256": file_sha256(h5_path(stem)),
        "kit_target": target,
        "kit_variant": variant,
        "kit_strip": "alpha_s/2pi per loop",
        "kit_layout": 1,
    } | ({"kit_hard_val_frac": HARD_VAL_FRAC} if variant == HARD else {})


def group_rows(g, target, meta):
    """FA pool rows (N, 5n+1) float64 of one HDF5 group: momenta reordered (beam_perm), PDG ids, stripped target."""
    parts = meta["particles"]
    perm = beam_perm([p["pdg"] for p in parts])
    mom = np.asarray(g["momenta"][:], dtype=np.float64)[:, perm]
    # a group without the target (new events to predict): 1, whose prediction residual is then the prediction itself
    y = (np.asarray(g[target][:], dtype=np.float64) if target in g else np.ones(len(g["momenta"]))) \
        / strip_factor(meta, target)
    n, P = mom.shape[0], mom.shape[1]
    pdg = np.broadcast_to(np.asarray([parts[i]["pdg"] for i in perm], dtype=np.float64), (n, P))
    return np.concatenate([mom.reshape(n, 4 * P), pdg, y[:, None]], axis=1)


def _hard_part(g, target, role):
    """A hard_train stratum's train part (its first 1 - HARD_VAL_FRAC) or val part (the rest): with `__hard`, the
    checkpoint is chosen on bulk val plus these held-out hard points, so the selection sees the fit protocol B tests"""
    m = g["momenta"].shape[0]
    cut = m - int(round(HARD_VAL_FRAC * m))
    sl = slice(0, cut) if role == "train" else slice(cut, m)
    return {k: g[k][sl] for k in ("momenta", target)}


def role_rows(name, role, n_events):
    """The rows of one role of a KIT dataset (module docstring)."""
    import h5py
    stem, target, variant = parse(name)
    n = int(n_events)
    with h5py.File(h5_path(stem), "r") as f:
        meta = json.loads(f.attrs["metadata"])
        bulk = f["bulk_train/bulk"]
        N = bulk["momenta"].shape[0]
        if role == "train":
            assert n <= N, f"{name}: {n} train events asked, bulk_train holds {N}"
            rows = group_rows({k: bulk[k][:n] for k in ("momenta", target)}, target, meta)
            if variant == HARD:
                rows = np.concatenate([rows] + [group_rows(_hard_part(f[f"hard_train/{st}"], target, "train"),
                                                           target, meta) for st in sorted(f["hard_train"])], axis=0)
            return rows
        if role == "val":
            assert n <= N, f"{name}: {n} val events asked, bulk_train holds {N}"
            rows = group_rows({k: bulk[k][N - n:] for k in ("momenta", target)}, target, meta)
            if variant == HARD:
                rows = np.concatenate([rows] + [group_rows(_hard_part(f[f"hard_train/{st}"], target, "val"),
                                                           target, meta) for st in sorted(f["hard_train"])], axis=0)
            return rows
        assert role == "test", role
        g = f["bulk_test/bulk"]
        assert n <= g["momenta"].shape[0], f"{name}: {n} test events asked, bulk_test holds {g['momenta'].shape[0]}"
        return group_rows({k: g[k][:n] for k in ("momenta", target)}, target, meta)


def check_disjoint(name, n_train, n_val):
    """train (the first n_train of bulk_train) and val (its last n_val) share no event: refused otherwise (called by
    mg5_pipeline_final.register_recipe_processes for every KIT spec)."""
    import h5py
    stem, _, _ = parse(name)
    with h5py.File(h5_path(stem), "r") as f:
        N = f["bulk_train/bulk/momenta"].shape[0]
    assert n_train + n_val <= N, f"{name}: n_train {n_train} + n_val {n_val} > bulk_train {N}: train and val overlap"


def pool_path(rec, out_dir):
    """A KIT pool's path: its event count in the name, so pools of one dataset at different sizes never share a file
    (two runs of a data-size scan at once)"""
    return (f"{out_dir}/{rec['process']}_{rec['sqrts_min']:.0f}-{rec['sqrts_max']:.0f}GeV"
            f"_{rec['role']}_n{int(rec['n_events'])}_kit_amplitudes.npy")


def build_pool(rec, out):
    """Write the pool of recipe `rec` at `out` (atomically: concurrent trials of one size build identical bytes)."""
    rows = role_rows(rec["process"], rec["role"], rec["n_events"])
    tmp = f"{out}.tmp{os.getpid()}.npy"
    np.save(tmp, rows)
    os.replace(tmp, out)
    return out


def ensure_alphas(rec, out):
    """The pool's per-event alpha_s sidecar <pool>.alphas.npy (what data.target_physics_orders divides the coupling
    powers by, as the catalog's pools carry): the file's fixed alpha_s for every event. Written when missing."""
    ap = out[:-len(".npy")] + ".alphas.npy"
    if os.path.exists(ap):
        return ap
    stem, _, _ = parse(rec["process"])
    n = np.load(out, mmap_mode="r").shape[0]
    tmp = f"{ap[:-len('.npy')]}.tmp{os.getpid()}.npy"
    np.save(tmp, np.full(n, float(metadata(stem)["alpha_s"]), dtype=np.float64))
    os.replace(tmp, ap)
    return ap


def write_born(rows_born, out):
    """<pool>.born.npy and its identity record <pool>.born.npy.json (what data.target_over_born reads,
    experiment._born_of_pool): the exact Born of the pool's events in its row order, from the file's `born` target."""
    from keyed_aug import pool_identity
    bp = out[:-len(".npy")] + ".born.npy"
    b = np.asarray(rows_born, dtype=np.float64)
    assert np.all(np.isfinite(b)) and np.all(b > 0), f"{out}: a non-positive Born"
    tmp = f"{bp[:-len('.npy')]}.tmp{os.getpid()}.npy"
    np.save(tmp, b)
    os.replace(tmp, bp)
    json.dump({"pool_identity": list(pool_identity(out))}, open(bp + ".json", "w"))
    return bp


def ensure_born(rec, out):
    """The Born sidecar of a KIT pool whose target is a one-loop term of a file that carries `born` (ee_tt's
    virt_finite): the same events' Born (role_rows of the born target, same slicing). Written when missing or when
    its identity record names another pool; None for a file or target without one."""
    stem, target, variant = parse(rec["process"])
    meta = metadata(stem)
    if target == "born" or "born" not in meta["targets"] or target_info(meta, target)[0] == 0:
        return None
    bp = out[:-len(".npy")] + ".born.npy"
    if os.path.exists(bp + ".json"):
        from keyed_aug import pool_identity
        if list(json.load(open(bp + ".json"))["pool_identity"]) == list(pool_identity(out)):
            return bp
    name = f"{PREFIX}{stem}__born" + (f"__{variant}" if variant else "")
    return write_born(role_rows(name, rec["role"], rec["n_events"])[:, -1] * strip_factor(meta, "born"), out)


def stratum_rows(stem, target, split, stratum, path=None):
    """The FA pool rows of one stratum (for scoring); `path` another file in the benchmark's layout (default its own)."""
    import h5py
    with h5py.File(path or h5_path(stem), "r") as f:
        meta = json.loads(f.attrs["metadata"])
        return group_rows(f[f"{split}/{stratum}"], target, meta)


def strata(stem, split, path=None):
    import h5py
    with h5py.File(path or h5_path(stem), "r") as f:
        return sorted(f[split]) if split in f else []


def log10_name(stem, target, path=None):
    """The truth's log10 dataset name in a group: log10_<target> or log10_abs_<target>."""
    import h5py
    with h5py.File(path or h5_path(stem), "r") as f:
        g = f["bulk_test/bulk"] if "bulk_test/bulk" in f else {}
        for k in (f"log10_{target}", f"log10_abs_{target}"):
            if k in g:
                return k
    if path:                 # a file without truth: the benchmark file's name
        return log10_name(stem, target)
    raise KeyError(f"{stem}: no log10 dataset for {target}")


def signs(stems=None):
    """{file/target/split/stratum: (events, events with target <= 0)} over every group of every file: whether a
    target is signed where it is scored (tools/kit_predict.py reads positive targets only)."""
    import glob
    import h5py
    stems = stems or sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(kit_dir(), "*.h5")))
    out = {}
    for stem in stems:
        meta = metadata(stem)
        with h5py.File(h5_path(stem), "r") as f:
            for split in SPLITS:
                for st in (sorted(f[split]) if split in f else []):
                    for target in meta["targets"]:
                        if target in f[split][st]:
                            y = np.asarray(f[split][st][target][:])
                            out[f"{stem}/{target}/{split}/{st}"] = (int(y.size), int((y <= 0).sum()))
    return out


if __name__ == "__main__":
    # python tools/kit_hardps.py signs [<file> ...]: per group, the events and those with a non-positive target
    if len(sys.argv) >= 2 and sys.argv[1] == "signs":
        res = signs(sys.argv[2:] or None)
        for k, (n, neg) in res.items():
            print(f"{k}: {n} events, {neg} with target <= 0")
        print(f"SIGNS total {sum(n for n, _ in res.values())} events, {sum(m for _, m in res.values())} non-positive")
    else:
        sys.exit("usage: python tools/kit_hardps.py signs [<file> ...]")
