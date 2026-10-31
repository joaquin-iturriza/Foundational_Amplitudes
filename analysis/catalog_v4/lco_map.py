"""The leave-class-out read of the held-out multiplicity gap (plan v3_prep_explore, design of 2026-10-31, stage 2).

Is the extra per-event residual of catalog_v4's held-out multiplicity pools shared across held-out processes? Stage 1
(dipole_residual.py --prop --dump) wrote per probe 500 events (seed 0) of the leg-order-free features (lof_features)
and the control's zero-shot residual u. Here, per arm (seed) of the control pair:
  deciding read   the H2 pools' classes (census_classes.json) in 10 folds grouped by class (GroupKFold over the sorted
                  pools); each fold's pools scored by a gradient-boosted map of u (HistGradientBoostingRegressor, 300
                  iterations, defaults, random_state 0) fitted on the other folds' H2 events only;
                  f = median_p removed_p / median_p gap_p, gap_p = var(u)_p - floor of p's cell (median var(u) over
                  the cell's TR pools), removed_p = var(u)_p - var(u - u_hat)_p (the shape-gap rule's per-cell floor)
  first           f_ceil: removed_p = R2_own,p var(u)_p, the stage-1 own-label ceiling (K = 1600 of 2000 events); the
                  design's thresholds are shares of it
  beside it       the same folds on the factors t, logF, log P, log D only, and without each feature group (drop-group)
  diagnostics     leave-one-cell-out (a cell's pools scored by the map of the other cells' H2 pools); the training-pool
                  map (fitted on every TR pool, applied to H2) and its TR score, 5 folds grouped by pool; the own-label
                  pools with a final gluon or photon (log D column nonzero) against those without; per cell medians
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
    """{probe: (X, u, own-label R^2 on all the probe's events)}, names. var(u), the gap and the floors are taken over
    the 500 dumped events (removed = var(u) - var(u - u_hat) must use one sample); the stored var_u is not used."""
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
        if not tr:          # one group only (a dry run's single cell): nothing to fit on
            return None
        m = fit(np.concatenate([pools[p][0][:, cols] for p in tr]), np.concatenate([pools[p][1] for p in tr]))
        for p in pools:
            if groups[p] == g:
                res[p] = float(np.var(pools[p][1] - m.predict(pools[p][0][:, cols])))
    return res


def f_of(vh, vr, floor):
    """median removed over median gap, each H2 pool against its own cell's floor."""
    vh, vr, floor = map(np.asarray, (vh, vr, floor))
    return float(np.median(vh - vr) / np.median(vh - floor))


def floors(vt_by_cell):
    return {c: float(np.median(v)) for c, v in vt_by_cell.items()}


def boot(vh, vr, cells, vt_by_cell):
    """16-84% of f over H2 pools and each cell's TR pools resampled with replacement, 1000 draws, seed 0 for every
    read, so the reads' intervals are paired (the same resamples)."""
    rng = np.random.default_rng(0)
    vh, vr = np.asarray(vh), np.asarray(vr)
    fs = []
    for _ in range(1000):
        i = rng.integers(len(vh), size=len(vh))
        fl = floors({c: np.asarray(v)[rng.integers(len(v), size=len(v))] for c, v in vt_by_cell.items()})
        fs.append(f_of(vh[i], vr[i], [fl[cells[k]] for k in i]))
    return [round(float(np.quantile(fs, 0.16)), 3), round(float(np.quantile(fs, 0.84)), 3)]


def verdict(out, arms):
    """The design's rule (passed 2026-10-31). Per seed, margin = max(seed difference, f_lco's bootstrap half-range):
    the route ends when f_ceil < 0.1 on both seeds; yes when f_lco - margin >= 0.5 f_ceil and >= 0.1; no when
    f_lco + margin < 0.25 f_ceil or < 0.05; else between. A branch holds only on both seeds (else between). On yes,
    the next arm: factors-only f >= f_lco / 2 -> data.target_propagators; else f_lco held by the pools with a final
    gluon or photon (f_lco_emitter > f_lco_no_emitter) -> the parent-factorised target; else the feature group whose
    drop loses the most f. Flags: cell-local (leave-one-cell-out < 0.25 f_ceil on a yes), f_lco > f_ceil (reported,
    not a bug: the map sees about 98k events of other pools), the training-pool map's TR score not near 0 (|r2| > 0.02)."""
    V = {"per_seed": {}, "flags": []}
    if any(out[a].get("f_lco") is None for a in arms):
        V["branch"] = "no read (a fold or cell had nothing to fit on)"
        return V
    if all(out[a]["f_ceil"] < 0.1 for a in arms):
        V["branch"] = "route ends: f_ceil < 0.1 on every seed (the gap is level and spread, not per-event shape)"
        return V
    sd = out.get("seed_difference_f_lco") or 0.0
    for a in arms:
        R = out[a]
        f, fc = R["f_lco"], R["f_ceil"]
        m = max(sd, (R["boot_lco"][1] - R["boot_lco"][0]) / 2)
        b = ("yes" if f - m >= 0.5 * fc and f - m >= 0.1 else
             "no" if f + m < 0.25 * fc or f + m < 0.05 else "between")
        V["per_seed"][a] = {"branch": b, "f_lco": f, "f_ceil": fc, "margin": round(m, 4)}
        if f > fc:
            V["flags"].append(f"{a}: f_lco {f} > f_ceil {fc}")
        if abs(R["median_r2_trainmap_TR_oof"]) > 0.02:
            V["flags"].append(f"{a}: training-pool map's TR score {R['median_r2_trainmap_TR_oof']} not near 0")
        if b == "yes":
            drops = {g: R.get(f"f_lco_drop_{g}") for g in GROUPS if R.get(f"f_lco_drop_{g}") is not None}
            V["per_seed"][a]["next_arm"] = (
                "data.target_propagators" if R["f_lco_factors"] >= f / 2 else
                "parent-factorised target" if R.get("f_lco_emitter", -1) > R.get("f_lco_no_emitter", -1) else
                f"the {min(drops, key=drops.get)} group's term" if drops else "unnamed")
            if R.get("f_loco_cell") is not None and R["f_loco_cell"] < 0.25 * fc:
                V["flags"].append(f"{a}: cell-local (leave-one-cell-out {R['f_loco_cell']})")
    bs = {v["branch"] for v in V["per_seed"].values()}
    V["branch"] = bs.pop() if len(bs) == 1 else "between (seeds disagree)"
    return V


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
        H, names = load(os.path.join(a.root, f"{arm}_H2"))
        T, _ = load(os.path.join(a.root, f"{arm}_TR"))
        cells = {cell(p) for p in H}
        T = {p: v for p, v in T.items() if cell(p) in cells}
        hp = sorted(H)
        vh = {p: float(np.var(H[p][1])) for p in hp}
        vt_by_cell = {}
        for p, v in T.items():
            vt_by_cell.setdefault(cell(p), []).append(float(np.var(v[1])))
        fl = floors(vt_by_cell)
        bare = sorted(c for c in cells if c not in fl)
        if bare:
            raise SystemExit(f"lco_map: held-out cells with no training pool, no floor: {bare} ({arm})")
        hc = [cell(p) for p in hp]
        v1 = [vh[p] for p in hp]
        f1 = [fl[c] for c in hc]
        iD = names.index("logD")
        emit = np.array([bool(np.any(H[p][0][:, iD] != 0)) for p in hp])
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
        R = {"n_H2": len(hp), "n_TR": len(T), "n_classes": len(set(klass)), "n_H2_emitter": int(emit.sum()),
             "median_var_H2": float(np.median(v1)), "floor_TR_per_cell": fl,
             "median_gap": float(np.median(np.array(v1) - np.array(f1))),
             "median_ceiling_own_K1600": float(np.nanmedian([H[p][2] for p in hp]))}
        vc = [vh[p] * (1 - H[p][2]) for p in hp]           # first: the own-label ceiling's f
        R["f_ceil"] = round(f_of(v1, vc, f1), 4)
        R["boot_ceil"] = boot(v1, vc, hc, vt_by_cell)
        print("LCO_CEIL", json.dumps({arm: R["f_ceil"]}), flush=True)
        per = {}
        for key, (grp, cols) in reads.items():
            vr = scored(H, grp, cols)
            if vr is None:
                R[f"f_{key}"] = None
                continue
            per[key] = vr
            v2 = [vr[p] for p in hp]
            R[f"f_{key}"] = round(f_of(v1, v2, f1), 4)
            R[f"boot_{key}"] = boot(v1, v2, hc, vt_by_cell)
            if key == "lco":
                for lab, m in (("emitter", emit), ("no_emitter", ~emit)):
                    if m.any():
                        R[f"f_lco_{lab}"] = round(f_of(np.array(v1)[m], np.array(v2)[m], np.array(f1)[m]), 4)
            R[f"median_r2_{key}"] = round(float(np.median([1 - vr[p] / vh[p] for p in hp])), 4)
        # the training-pool map: fitted on every TR pool, applied to H2; its own TR score grouped 5-fold by pool
        m = fit(np.concatenate([v[0] for v in T.values()]), np.concatenate([v[1] for v in T.values()]))
        vr = {p: float(np.var(H[p][1] - m.predict(H[p][0]))) for p in hp}
        R["f_trainmap"] = round(f_of(v1, [vr[p] for p in hp], f1), 4)
        tp = sorted(T)
        tfold = {p: i % min(5, len(tp)) for i, p in enumerate(tp)}
        vtr = scored(T, tfold, allc)
        R["median_r2_trainmap_TR_oof"] = round(float(np.median([1 - vtr[p] / np.var(T[p][1]) for p in tp])), 4)
        R["per_cell"] = {c: {"n_H2": sum(cell(p) == c for p in hp),
                             **{f"median_r2_{k}": round(float(np.median([1 - per[k][p] / vh[p] for p in hp
                                                                          if cell(p) == c])), 4)
                                for k in ("lco", "lco_factors") if k in per}}
                         for c in sorted(cells)}
        out[arm] = R
        print("LCO_MAP", json.dumps({arm: {k: v for k, v in R.items() if k != "per_cell"}}), flush=True)
    s = [out[x]["f_lco"] for x in a.arms]
    out["seed_difference_f_lco"] = round(float(max(s) - min(s)), 4) if None not in s else None
    out["verdict"] = verdict(out, a.arms)
    print("LCO_VERDICT", json.dumps(out["verdict"]), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
