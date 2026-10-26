"""catalog_v4's training density against the 13 TeV pp events' (catalog_coverage, CPU only, no new label).

The coverage check before catalog_v4 was generated (analysis/catalog_v3/coverage_check.py) measured containment: which
share of a class's pp events lies inside the new cuts and window. It did not measure density: how many of catalog_v4's
training events fall where the pp events sit. Its sqrt(s_hat) "effective sample size" was (sum w)^2 / (n sum w^2) over
the pp events with w = p_pp / q, which is 1 for any q proportional to p_pp on pp's support, whatever the magnitude of
q there: it measured the shape of the ratio, not how thin q is. This script measures the density.

Per hadronic tree class (the 1396 classes with a 13 TeV pp pool, recipes/catalog_v3_pp_test.yaml), four samples, every
event in the partonic rest frame (the pp events boosted along z, as pp_compare.support_masks):
  v4     catalog_v4's training pool of the class (its own, or the catalog_v4 class holding its tree after the merges and
         splits of data/class_changes.json), the first N_POOL rows (a pool is a concatenation of chunks, each a full
         mixture shuffled, so the first rows are the mixture)
  v3     catalog_v3's training pool of the class, the first N_POOL rows (the control: the pp-trained arm's gain over the
         window-trained arm is large, so a density measure that means anything must follow it there)
  pp     the class's 13 TeV pp training pool (MadEvent's unweighted events, 1e4): pp's density
  ppval  the first 2000 rows of its pp validation pool: the events every pp read scored
Three variables, binned jointly:
  s      log10 sqrt(s_hat) / GeV
  u      log(1/(1 - |cos theta|)) of the most forward final particle (coverage_check.u_forward)
  y      log10 of the smallest 2|p_i.p_j| / s_hat over every pair of external legs but the two beams: a final pair
         collinear, or a final particle collinear to a beam (small |t|), whichever is nearest
and one alone:
  pt     log10 of the smallest final-particle pT / GeV
    site run --item catalog_coverage ccin2p3 FA -- python analysis/catalog_v4/density_check.py run \
        > analysis/catalog_v4/data/density_hists.json                                 (login node, ~10 min)
    python3 analysis/catalog_v4/density_check.py      -> data/density_check.json, figs/density_check.{png,pdf}
The summary's method and its own choices are in summary()'s docstring.
"""
import argparse, json, os, sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

N_POOL = 20000
# fine bins written by `run`; the summary coarsens them
S_EDGES = np.round(np.arange(1.0, 3.80001, 0.1), 3)          # 10 GeV .. 6.3 TeV, 28 bins
U_EDGES = np.arange(0.0, 10.0001, 1.0)                        # |eta| < 5 is u = 9.31, 10 bins
Y_EDGES = np.arange(-8.0, 1.0001, 1.0)                        # 9 bins
PT_EDGES = np.round(np.arange(0.0, 3.80001, 0.1), 3)          # 1 GeV .. 6.3 TeV, 38 bins
SHAPE = (len(S_EDGES) - 1, len(U_EDGES) - 1, len(Y_EDGES) - 1)


def pools():
    """pp class -> (catalog_v4 training class or None, catalog_v3 training class or None)"""
    names = [e["name"] for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v3_pp_test.yaml")))["processes"]]
    tr4 = {e["name"]: e for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v4_train.yaml")))["processes"]}
    tr3 = {e["name"]: e for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v3_train.yaml")))["processes"]}
    cc = json.load(open(os.path.join(HERE, "data", "class_changes.json")))
    holder = {}                                       # catalog_v4 member -> the class whose pool holds it
    for g in cc["merges"]:
        for x in g["members"]:
            holder.setdefault(x, g["v4_class"])
    for sp in cc["splits"]:
        for k, v in sp["v4_classes"].items():
            for x in v["members"]:
                holder.setdefault(x, k)
    out = {}
    for n in names:
        x = n[len("pp13_"):]
        c4 = "v4_" + x
        c4 = c4 if c4 in tr4 else holder.get(c4)
        out[n] = {"v4": (c4, tr4[c4]["sqrts"]) if c4 in tr4 else None,
                  "v3": ("v3_" + x, tr3["v3_" + x]["sqrts"]) if "v3_" + x in tr3 else None}
    return out


def features(a, k, boost):
    """rows -> (s, u, y, pt) per event"""
    P = np.array(a[:, :4 * k], dtype=np.float64).reshape(-1, k, 4)
    tot = P[:, 0] + P[:, 1]
    if boost:
        bz = tot[:, 3] / tot[:, 0]
        g = 1.0 / np.sqrt(1.0 - bz ** 2)
        E, pz = P[..., 0].copy(), P[..., 3].copy()
        P[..., 0] = g[:, None] * (E - bz[:, None] * pz)
        P[..., 3] = g[:, None] * (pz - bz[:, None] * E)
    sh = np.clip(tot[:, 0] ** 2 - (tot[:, 1:] ** 2).sum(-1), 1e-300, None)
    p = P[:, 2:, 1:]
    c = np.abs(p[..., 2]) / np.clip(np.linalg.norm(p, axis=-1), 1e-300, None)
    u = np.log(1.0 / np.clip(1.0 - c.max(1), 1e-300, None))
    dot = lambda x, z: x[..., 0] * z[..., 0] - (x[..., 1:] * z[..., 1:]).sum(-1)
    ys = [2.0 * np.abs(dot(P[:, i], P[:, j])) / sh for i in range(k) for j in range(i + 1, k) if (i, j) != (0, 1)]
    y = np.log10(np.clip(np.min(ys, axis=0), 1e-300, None))
    pt = np.log10(np.clip(np.hypot(P[:, 2:, 1], P[:, 2:, 2]).min(1), 1e-300, None))
    return 0.5 * np.log10(sh), u, y, pt


def bin_of(x, e):
    return np.clip(np.searchsorted(e, x, side="right") - 1, 0, len(e) - 2)


def hists(a, k, boost):
    s, u, y, pt = features(a, k, boost)
    j = np.ravel_multi_index((bin_of(s, S_EDGES), bin_of(u, U_EDGES), bin_of(y, Y_EDGES)), SHAPE)
    jc = np.bincount(j, minlength=int(np.prod(SHAPE)))
    nz = np.nonzero(jc)[0]
    return {"n": int(len(s)), "joint": [[int(i), int(jc[i])] for i in nz],
            "pt": [int(x) for x in np.bincount(bin_of(pt, PT_EDGES), minlength=len(PT_EDGES) - 1)]}


def one(task):
    n, P = task
    import mg5_pipeline_final as mg
    D, C = os.environ["DATA_DIR"], os.path.join(os.environ["SCRATCH"], "amp_data_cache")
    row = {"name": n}
    try:
        k = len(mg.PROCESSES[n]["pdg_ids"])
        row["nfinal"] = k - 2
        row["pp"] = hists(np.load(os.path.join(C, f"{n}_13000-13000GeV_train_lhe_amplitudes.npy"), mmap_mode="r"), k, True)
        row["ppval"] = hists(np.load(os.path.join(D, f"{n}_13000-13000GeV_val_lhe_amplitudes.npy"),
                                     mmap_mode="r")[:2000], k, True)
        for cat in ("v4", "v3"):
            if P[cat] is None:
                continue
            c, (lo, hi) = P[cat]
            a = np.load(os.path.join(C, f"{c}_{int(lo)}-{int(hi)}GeV_train_smix_amplitudes.npy"), mmap_mode="r")
            kc = len(mg.PROCESSES[c]["pdg_ids"])
            assert kc == k, (c, kc, k)
            row[cat] = dict(hists(a[:N_POOL], k, False), pool=c, sqrts=[float(lo), float(hi)], n_pool_rows=int(a.shape[0]))
    except Exception as e:                   # a class that fails is listed with its error, never dropped silently
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def run(workers, only=None):
    from multiprocessing import Pool
    P = pools()
    if only:
        P = {n: P[n] for n in sorted(P)[:only]}
    rows = {}
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, sorted(P.items()), chunksize=4)):
            rows[r["name"]] = r
            if (i + 1) % 100 == 0:
                print(f"[{i + 1}/{len(P)}]", file=sys.stderr, flush=True)
    err = {n: r["error"] for n, r in rows.items() if "error" in r}
    print(f"RUN_SUMMARY rows={len(rows)} errors={len(err)} {list(err.items())[:5]}", file=sys.stderr, flush=True)
    json.dump({"edges": {"s": S_EDGES.tolist(), "u": U_EDGES.tolist(), "y": Y_EDGES.tolist(), "pt": PT_EDGES.tolist()},
               "n_pool": N_POOL, "rows": rows}, sys.stdout)


def summary():
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="summary", choices=("run", "summary"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", type=int, default=None, help="the first N classes only (a test)")
    a = ap.parse_args()
    if a.mode == "run":
        run(a.workers, a.only)
    else:
        summary()
