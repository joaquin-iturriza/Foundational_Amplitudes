"""A per-process level and spread correction from K labelled events on the held-out multiplicity pools
(plan v3_prep_explore, option (c) of step 6; CPU, no new run).

par:v4_lco_not ended the event-level prior route: nothing known at prediction time predicts the held-out multiplicity
pools' residual event by event, and given the event's own target t a map undoes the network's compression of the
pools' target range. A compression is a spread error of the whole process: t ~ a + b q with b != 1. Here, per pool,
K labelled events fit a correction of the control's zero-shot prediction q, scored on the pool's other events.

Data: the control pair's zero-shot reads (v4_dimensionless_muon, _seed2; best checkpoint, 2000 validation events per
pool; analysis/catalog_v3/data/short_reads/<run>/{H2,TR}/<probe>.npz with q and t, as kshot_level.py), the 218 held-out
multiplicity pools (H2) and the training pools (TR) of their cells (n, initial state; data/tree_levels.json).
u = q - t. Per pool and K, NDRAW draws of K events (rng per pool, seed 0, shared by every method: paired); the fit
uses only the K drawn events and the prior pools, the score only the other 2000 - K events (asserted disjoint):
  off     t_hat = q + mean_K(t - q)                                   (the plain offset, kshot_level.py's off<K>)
  aff     t_hat = a + b q, ordinary least squares on the K events    (level and spread)
  shr_tr  aff with b shrunk to the slope prior of the pool's cell's training pools: b = (Sqt + lam b0)/(Sqq + lam),
          a = mean_K t - b mean_K q; b0 = median full-data slope over the cell's TR pools, lam = s2 / tau2 with s2 the
          median residual variance after their affine fit and tau2 the variance of their slopes (the deployable one:
          the prior uses training data only)
  shr_ho  the same with the prior from the H2 pools of the other classes (census_classes.json), leave-class-out
          (the precedent of kshot_level.py's shrunk offset, which pulls toward the other probes of the axis)
  ceil    the affine's ceiling: 5 folds of the 2000 events, fit on 1600, scored on the other 400; ceil_cubic the
          same for a cubic in q (diagnostic: any smooth function of q per process)
mse_m = mean over draws of the scored MSE. Share of the gap removed (par:v4_lco's f):
  f_m = median_p [var(u)_p - mse_m,p] / median_p [var(u)_p - floor_c(p)], floor_c the median var(u) over the cell's
  TR pools; var(u) is what an exact offset reaches, so f_m is what the correction adds beyond the level.
Margins: the 16-84% range of f over H2 pools and each cell's TR pools resampled (1000 draws, seed 0, paired across
methods). The control: the same corrections on the TR pools (median mse_m / var(u)), so a correction that helps every
pool alike is told from one that closes the held-out gap. Diagnostics: the median slope b over each set, the shares
with b > 1 (a compressed prediction).
    python3 analysis/catalog_v4/kshot_spread.py --check   -> slopes and the ceiling only (the design's check)
    python3 analysis/catalog_v4/kshot_spread.py --out analysis/catalog_v4/data/kshot_spread.json
The last line printed is the JSON of the numbers docs/results.tex quotes.
"""
import argparse
import json
import os

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lco_map import boot, f_of  # noqa: E402  (par:v4_lco's f and its paired bootstrap)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SR = os.path.join(ROOT, "analysis", "catalog_v3", "data", "short_reads")
ARMS = {"v4_dimensionless_muon": "s42", "v4_dimensionless_muon_seed2": "s43"}
KS = (4, 16, 64, 256)
NDRAW = 200
METHODS = ("off", "aff", "shr_tr", "shr_ho")


def load(run, axis, keep=None):
    out = {}
    d = os.path.join(SR, run, axis)
    for f in sorted(os.listdir(d)):
        p = f[:-4]
        if not f.endswith(".npz") or (keep is not None and not keep(p)):
            continue
        z = np.load(os.path.join(d, f))
        if "q" not in z.files or len(z["q"]) != len(z["t"]) or not len(z["t"]):
            continue
        out[p] = (z["q"].astype(np.float64), z["t"].astype(np.float64))
    return out


def slope(q, t):
    qc = q - q.mean()
    b = float(qc @ (t - t.mean()) / (qc @ qc))
    return b, float(np.var(t - t.mean() - b * qc))


def prior(pools):
    """(b0, lam) from full-data affine fits of the given pools: median slope, s2 / tau2."""
    bs, s2 = zip(*(slope(q, t) for q, t in pools))
    return float(np.median(bs)), float(np.median(s2)) / (float(np.var(bs)) + 1e-9)


def fit_apply(q, t, pick, rest, b0=None, lam=0.0):
    qk, tk = q[pick], t[pick]
    qc = qk - qk.mean()
    sqq, sqt = float(qc @ qc), float(qc @ (tk - tk.mean()))
    if b0 is None:
        b = sqt / sqq if sqq > 0 else 1.0
    else:
        b = (sqt + lam * b0) / (sqq + lam)
    a = tk.mean() - b * qk.mean()
    return float(np.mean((t[rest] - a - b * q[rest]) ** 2))


def ceiling(q, t, deg=1):
    """a polynomial of degree deg in q (1: the affine) fitted on 1600 events, scored on the other 400, 5 folds."""
    idx = np.random.default_rng(0).permutation(len(q))
    fo = np.array_split(idx, 5)
    z = (q - q.mean()) / q.std()
    A = np.stack([z ** k for k in range(deg + 1)], 1)
    out = []
    for i in range(5):
        tr = np.concatenate([fo[j] for j in range(5) if j != i])
        c, *_ = np.linalg.lstsq(A[tr], t[tr], rcond=None)
        out.append(float(np.mean((t[fo[i]] - A[fo[i]] @ c) ** 2)))
    return float(np.mean(out))


def per_pool(q, t, k, seed, priors):
    rng = np.random.default_rng([0, seed])
    n = len(q)
    acc = {m: 0.0 for m in METHODS}
    for _ in range(NDRAW):
        pick = rng.choice(n, k, replace=False)
        rest = np.setdiff1d(np.arange(n), pick)
        assert len(rest) + k == n and not np.intersect1d(pick, rest).size   # fit and score events disjoint
        acc["off"] += float(np.mean((t[rest] - q[rest] - (t[pick] - q[pick]).mean()) ** 2))
        acc["aff"] += fit_apply(q, t, pick, rest)
        for m in ("shr_tr", "shr_ho"):
            if priors.get(m) is not None:
                acc[m] += fit_apply(q, t, pick, rest, *priors[m])
    return {m: v / NDRAW for m, v in acc.items() if m in ("off", "aff") or priors.get(m) is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="slopes and the affine ceiling only")
    ap.add_argument("--out")
    a = ap.parse_args()
    tl = json.load(open(os.path.join(HERE, "data", "tree_levels.json")))
    info = dict(tl["train"])
    for ax in tl["heldout"].values():
        info.update(ax)
    cell = lambda p: f"{info[p]['n']}{info[p]['initial']}" if p in info else None
    cls = json.load(open(os.path.join(HERE, "census_classes.json")))["classes"]
    rep = {m: r for r, ms in cls.items() for m in ms}
    res = {}
    for run, tag in ARMS.items():
        H = load(run, "H2")
        cells = {cell(p) for p in H}
        T = load(run, "TR", keep=lambda p: cell(p) in cells)
        hp, tp = sorted(H), sorted(T)
        vh = {p: float(np.var(H[p][0] - H[p][1])) for p in hp}
        vt = {p: float(np.var(T[p][0] - T[p][1])) for p in tp}
        vt_by_cell = {}
        for p in tp:
            vt_by_cell.setdefault(cell(p), []).append(vt[p])
        fl = {c: float(np.median(v)) for c, v in vt_by_cell.items()}
        bare = sorted(c for c in cells if c not in fl)
        if bare:
            raise SystemExit(f"kshot_spread: held-out cells with no training pool: {bare} ({run})")
        hc = [cell(p) for p in hp]
        v1, f1 = [vh[p] for p in hp], [fl[c] for c in hc]
        bH = {p: slope(*H[p])[0] for p in hp}
        bT = {p: slope(*T[p])[0] for p in tp}
        R = {"n_H2": len(hp), "n_TR": len(tp), "median_var_u_H2": round(float(np.median(v1)), 4),
             "floor_TR_per_cell": {c: round(v, 4) for c, v in fl.items()},
             "median_gap": round(float(np.median(np.array(v1) - np.array(f1))), 4),
             "median_slope_H2": round(float(np.median(list(bH.values()))), 4),
             "median_slope_TR": round(float(np.median(list(bT.values()))), 4),
             "share_slope_gt1_H2": round(float(np.mean([b > 1 for b in bH.values()])), 3),
             "share_slope_gt1_TR": round(float(np.mean([b > 1 for b in bT.values()])), 3)}
        cH = {p: ceiling(*H[p]) for p in hp}
        cT = {p: ceiling(*T[p]) for p in tp}
        vc = [cH[p] for p in hp]
        R["f_ceil"] = round(f_of(v1, vc, f1), 4)
        R["boot_ceil"] = boot(v1, vc, hc, vt_by_cell)
        R["median_ratio_ceil_H2"] = round(float(np.median([cH[p] / vh[p] for p in hp])), 4)
        R["median_ratio_ceil_TR"] = round(float(np.median([cT[p] / vt[p] for p in tp])), 4)
        c3 = [ceiling(*H[p], deg=3) for p in hp]
        R["f_ceil_cubic"] = round(f_of(v1, c3, f1), 4)
        R["median_ratio_ceil_cubic_H2"] = round(float(np.median(np.array(c3) / np.array(v1))), 4)
        if not a.check:
            pri_tr = {c: prior([T[p] for p in tp if cell(p) == c]) for c in cells}
            klass = {p: rep.get(p, p) for p in hp}
            for K in KS:
                sh = {}
                for i, p in enumerate(hp):
                    oth = [H[o] for o in hp if klass[o] != klass[p]]
                    sh[p] = per_pool(*H[p], K, i, {"shr_tr": pri_tr[cell(p)], "shr_ho": prior(oth)})
                st = {}
                for i, p in enumerate(tp):
                    oth = [T[o] for o in tp if o != p and cell(o) == cell(p)]
                    st[p] = per_pool(*T[p], K, 10000 + i, {"shr_tr": prior(oth) if len(oth) > 1 else None})
                for m in METHODS:
                    v2 = [sh[p][m] for p in hp]
                    R[f"f_{m}{K}"] = round(f_of(v1, v2, f1), 4)
                    R[f"boot_{m}{K}"] = boot(v1, v2, hc, vt_by_cell)
                    R[f"median_mse_{m}{K}_H2"] = round(float(np.median(v2)), 4)
                    if all(m in st[p] for p in tp):
                        R[f"median_ratio_{m}{K}_TR"] = round(float(np.median([st[p][m] / vt[p] for p in tp])), 4)
                    R[f"median_ratio_{m}{K}_H2"] = round(float(np.median([sh[p][m] / vh[p] for p in hp])), 4)
                print("K", K, tag, json.dumps({m: R[f"f_{m}{K}"] for m in METHODS}), flush=True)
        res[tag] = R
    if not a.check:
        res["seed_difference"] = {f"{m}{K}": round(abs(res["s42"][f"f_{m}{K}"] - res["s43"][f"f_{m}{K}"]), 4)
                                  for m in METHODS for K in KS}
        res["seed_difference"]["ceil"] = round(abs(res["s42"]["f_ceil"] - res["s43"]["f_ceil"]), 4)
        res["verdict"] = verdict(res)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


def verdict(res):
    """The design's rule (primary: shr_tr at K = 16). Per seed, margin = max(seed difference, bootstrap half-range):
    ends when f_ceil < 0.1 on both seeds (no per-process affine carries the gap); yes when f - margin >= 0.5 f_ceil
    and >= 0.1; no when f + margin < 0.25 f_ceil or < 0.05; else between. A branch holds only on both seeds."""
    V = {"per_seed": {}}
    if all(res[s]["f_ceil"] < 0.1 for s in ("s42", "s43")):
        V["branch"] = "ends: f_ceil < 0.1 on both seeds"
        return V
    for s in ("s42", "s43"):
        R = res[s]
        f, fc = R["f_shr_tr16"], R["f_ceil"]
        m = max(res["seed_difference"]["shr_tr16"], (R["boot_shr_tr16"][1] - R["boot_shr_tr16"][0]) / 2)
        b = ("yes" if f - m >= 0.5 * fc and f - m >= 0.1 else
             "no" if f + m < 0.25 * fc or f + m < 0.05 else "between")
        V["per_seed"][s] = {"branch": b, "f": f, "f_ceil": fc, "margin": round(m, 4)}
    bs = {v["branch"] for v in V["per_seed"].values()}
    V["branch"] = bs.pop() if len(bs) == 1 else "between (seeds disagree)"
    return V


if __name__ == "__main__":
    main()
