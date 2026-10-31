"""The leave-class-out read of the held-out multiplicity gap (plan v3_prep_explore, design of 2026-10-31, stage 2).

Is the extra per-event residual of catalog_v4's held-out multiplicity pools shared across held-out processes? Stage 1
(dipole_residual.py --prop --dump) wrote per probe 500 events (seed 0) of the leg-order-free features (lof_features)
and the control's zero-shot residual u. Here, per arm (seed) of the control pair:
  deciding read   the H2 pools' classes (census_classes.json) in 10 folds grouped by class (GroupKFold over the sorted
                  pools); each fold's pools scored by a gradient-boosted map of u (HistGradientBoostingRegressor, 300
                  iterations, defaults, random_state 0) fitted on the other folds' H2 events only;
                  f = (median var(u)_H2 - median var(u - u_hat)_H2) / (median var(u)_H2 - median var(u)_TR)
  beside it       the same folds on the factors t, logF, log P, log D only, and without each feature group (drop-group)
  diagnostics     leave-one-cell-out (a cell's pools scored by the map of the other cells' H2 pools); the training-pool
                  map (fitted on every TR pool, applied to H2) and its TR score, 5 folds grouped by pool; the own-label
                  ceiling of stage 1 (K = 1600); per cell medians
  margins         the 16-84% range of f over H2 and TR pools resampled with replacement (1000 draws, seed 0)
TR pools are those of the H2 pools' cells (as dipole_gate.py). Writes one JSON.
    python3 analysis/catalog_v4/lco_map.py --root <dump root> --out <json> [--arms a b]
"""
import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("v4_dimensionless_muon", "v4_dimensionless_muon_seed2")
GROUPS = {"pairs": ("sij",), "energy_angle": ("lE", "cth", "lshat"), "species": ("n_",),
          "factors": ("t", "logF", "logP", "logD")}


def load(d):
    """{probe: (X, u, var_u over all the probe's events, own-label ceiling)}, names."""
    out, names = {}, None
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f)
        names = [str(x) for x in z["names"]]
        out[os.path.basename(f)[:-4]] = (z["X"].astype(np.float64), z["u"], float(z["ceiling_own"]))
    return out, names


def cols_of(names, group):
    pre = GROUPS[group]
    return [i for i, n in enumerate(names) if (n in pre if group == "factors" else n.startswith(pre))]


def fit(X, y):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, y)


def scored(pools, groups, cols):
    """{probe: var(u - u_hat)} with each group's pools predicted by the map fitted on the other groups' pools."""
    res = {}
    for g in sorted(set(groups.values())):
        tr = [p for p in pools if groups[p] != g]
        m = fit(np.concatenate([pools[p][0][:, cols] for p in tr]), np.concatenate([pools[p][1] for p in tr]))
        for p in pools:
            if groups[p] == g:
                res[p] = float(np.var(pools[p][1] - m.predict(pools[p][0][:, cols])))
    return res


def f_of(vh, vr, vt):
    return (np.median(vh) - np.median(vr)) / (np.median(vh) - np.median(vt))


def boot(vh, vr, vt, rng):
    vh, vr, vt = map(np.asarray, (vh, vr, vt))
    fs = []
    for _ in range(1000):
        i, j = rng.integers(len(vh), size=len(vh)), rng.integers(len(vt), size=len(vt))
        fs.append(f_of(vh[i], vr[i], vt[j]))
    return [round(float(np.quantile(fs, 0.16)), 3), round(float(np.quantile(fs, 0.84)), 3)]


def main():
    from sklearn.model_selection import GroupKFold
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="stage 1's dump root: <root>/<arm>_<axis>/<probe>.npz")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tl = json.load(open(os.path.join(HERE, "data", "tree_levels.json")))
    info = dict(tl["train"])
    for ax in tl["heldout"].values():
        info.update(ax)
    cell = lambda p: f"{info[p]['n']}{info[p]['initial']}" if p in info else None
    cls = json.load(open(os.path.join(HERE, "census_classes.json")))["classes"]
    rep = {m: r for r, ms in cls.items() for m in ms}
    out = {}
    for arm in a.arms:
        rng = np.random.default_rng(0)
        H, names = load(os.path.join(a.root, f"{arm}_H2"))
        T, _ = load(os.path.join(a.root, f"{arm}_TR"))
        cells = {cell(p) for p in H}
        T = {p: v for p, v in T.items() if cell(p) in cells}
        hp = sorted(H)
        vh = {p: float(np.var(H[p][1])) for p in hp}
        vt = [float(np.var(v[1])) for v in T.values()]
        klass = [rep.get(p, p) for p in hp]
        fold = {}
        for k, (_, te) in enumerate(GroupKFold(min(10, len(set(klass)))).split(hp, groups=klass)):
            for i in te:
                fold[hp[i]] = k
        allc = list(range(len(names)))
        reads = {"lco": (fold, allc), "lco_factors": (fold, cols_of(names, "factors"))}
        for g in GROUPS:
            reads[f"lco_drop_{g}"] = (fold, [i for i in allc if i not in cols_of(names, g)])
        reads["loco_cell"] = ({p: cell(p) for p in hp}, allc)
        R = {"n_H2": len(hp), "n_TR": len(T), "n_classes": len(set(klass)), "median_var_H2": float(np.median(list(vh.values()))),
             "median_var_TR": float(np.median(vt)), "median_ceiling_own_K1600": float(np.nanmedian([H[p][2] for p in hp]))}
        per = {}
        for key, (grp, cols) in reads.items():
            vr = scored(H, grp, cols)
            per[key] = vr
            v1 = [vh[p] for p in hp]
            v2 = [vr[p] for p in hp]
            R[f"f_{key}"] = round(float(f_of(v1, v2, vt)), 4)
            R[f"boot_{key}"] = boot(v1, v2, vt, rng)
            R[f"median_r2_{key}"] = round(float(np.median([1 - vr[p] / vh[p] for p in hp])), 4)
        # the training-pool map: fitted on every TR pool, applied to H2; its own TR score grouped 5-fold by pool
        m = fit(np.concatenate([v[0] for v in T.values()]), np.concatenate([v[1] for v in T.values()]))
        vr = {p: float(np.var(H[p][1] - m.predict(H[p][0]))) for p in hp}
        R["f_trainmap"] = round(float(f_of([vh[p] for p in hp], [vr[p] for p in hp], vt)), 4)
        tp = sorted(T)
        tfold = {p: i % min(5, len(tp)) for i, p in enumerate(tp)}
        vtr = scored(T, tfold, allc)
        R["median_r2_trainmap_TR_oof"] = round(float(np.median([1 - vtr[p] / np.var(T[p][1]) for p in tp])), 4)
        R["per_cell"] = {c: {"n_H2": sum(cell(p) == c for p in hp),
                             "median_r2_lco": round(float(np.median([1 - per["lco"][p] / vh[p] for p in hp if cell(p) == c])), 4),
                             "median_r2_lco_factors": round(float(np.median([1 - per["lco_factors"][p] / vh[p]
                                                                             for p in hp if cell(p) == c])), 4)}
                         for c in sorted(cells)}
        out[arm] = R
        print("LCO_MAP", json.dumps({arm: {k: v for k, v in R.items() if k != "per_cell"}}), flush=True)
    s = [out[x]["f_lco"] for x in a.arms]
    out["seed_difference_f_lco"] = round(float(max(s) - min(s)), 4)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
