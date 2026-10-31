"""Arm against control on the held-out multiplicity pools' residual spread (plan v3_prep_explore, option (b), D216).

Per (arm, control) pair of runs with zero-shot reads in analysis/catalog_v3/data/short_reads/<run>/{H2,TR}/ (2000
validation events per pool at the best checkpoint; q and t in log|M|^2 units up to the target factor, which cancels in
u = q - t, so no re-basing for a changed standardization is needed):
  r = median_p var(u)_arm,p / var(u)_ctrl,p over the H2 pools both read
  s = median_p [var(u)_ctrl,p - var(u)_arm,p] / median_p [var(u)_ctrl,p - floor_c(p)], floor_c the median var(u)
      over the control's TR pools of the cell (par:v4_lco's gap; a control without TR reads takes
      FLOOR_RUN's, recorded as floor_run)
  mse_ratio = median_p MSE_arm,p / MSE_ctrl,p (par:v4_signext's measure), and r per cell
  boot: 16-84% of r and s over H2 pools resampled (1000 draws, seed 0)
With several pairs: the mean and the standard deviation of r and s over pairs, and those of the pair means.
    python3 analysis/catalog_v4/multext_compare.py --pairs ARM:CTRL ... [--out json]
The signext pairs (v4_dimensionless_muon_more_signed[_seedN] : v4_dimensionless_muon[_seedN]) give the spread of
r and s under a training-set change of the same kind (the design's check).
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FLOOR_RUN = "v4_dimensionless_muon"
SR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "analysis", "catalog_v3", "data", "short_reads")


def reads(run, axis):
    d = os.path.join(SR, run, axis)
    out = {}
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.endswith(".npz"):
            z = np.load(os.path.join(d, f))
            if "q" in z.files and len(z["q"]) == len(z["t"]) and len(z["t"]):
                out[f[:-4]] = z["q"].astype(np.float64) - z["t"].astype(np.float64)
    return out


def compare(arm, ctrl, cell):
    A, C = reads(arm, "H2"), reads(ctrl, "H2")
    hp = sorted(set(A) & set(C))
    cells = {cell(p) for p in hp}
    frun = ctrl if reads(ctrl, "TR") else FLOOR_RUN       # seeds without training-pool reads: seed 42's floors
    T = {p: float(np.var(u)) for p, u in reads(frun, "TR").items() if cell(p) in cells}
    byc = {}
    for p, v in T.items():
        byc.setdefault(cell(p), []).append(v)
    fl = {c: float(np.median(v)) for c, v in byc.items()}
    va, vc = np.array([np.var(A[p]) for p in hp]), np.array([np.var(C[p]) for p in hp])
    ma, mc = np.array([np.mean(A[p] ** 2) for p in hp]), np.array([np.mean(C[p] ** 2) for p in hp])
    f = np.array([fl[cell(p)] for p in hp])
    rf = lambda i: float(np.median(va[i] / vc[i]))
    sf = lambda i: float(np.median(vc[i] - va[i]) / np.median(vc[i] - f[i]))
    al = np.arange(len(hp))
    rng = np.random.default_rng(0)
    bs = [rng.integers(len(hp), size=len(hp)) for _ in range(1000)]
    R = {"floor_run": frun, "n": len(hp), "n_arm_only": len(set(A) - set(C)), "n_ctrl_only": len(set(C) - set(A)),
         "r": round(rf(al), 4), "s": round(sf(al), 4), "mse_ratio": round(float(np.median(ma / mc)), 4),
         "boot_r": [round(float(np.quantile([rf(i) for i in bs], q)), 4) for q in (0.16, 0.84)],
         "boot_s": [round(float(np.quantile([sf(i) for i in bs], q)), 4) for q in (0.16, 0.84)],
         "median_gap_ctrl": round(float(np.median(vc - f)), 4),
         "r_per_cell": {c: round(float(np.median([va[i] / vc[i] for i in al if cell(hp[i]) == c])), 4)
                        for c in sorted(cells)}}
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="ARM:CTRL run names under short_reads")
    ap.add_argument("--out")
    a = ap.parse_args()
    tl = json.load(open(os.path.join(HERE, "data", "tree_levels.json")))
    info = dict(tl["train"])
    for ax in tl["heldout"].values():
        info.update(ax)
    cell = lambda p: f"{info[p]['n']}{info[p]['initial']}" if p in info else None
    res = {}
    for pr in a.pairs:
        arm, ctrl = pr.split(":")
        res[pr] = compare(arm, ctrl, cell)
        print(pr, json.dumps({k: res[pr][k] for k in ("n", "r", "s", "mse_ratio", "boot_r", "boot_s")}), flush=True)
    if len(a.pairs) > 1:
        for k in ("r", "s", "mse_ratio"):
            x = np.array([res[p][k] for p in a.pairs])
            res[f"mean_{k}"] = round(float(x.mean()), 4)
            res[f"sd_{k}"] = round(float(x.std(ddof=1)), 4)
            res[f"se_mean_{k}"] = round(float(x.std(ddof=1) / np.sqrt(len(x))), 4)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
