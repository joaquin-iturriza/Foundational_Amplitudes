"""D83/D99: does a target between the dimensionless one and the emission scales keep the 2->4 gain without the cost on
the 43 new processes? (par:dimless_qperp: the emission scales take 2->4 from 52 to 2.2 and the 43 new from 0.48 to
0.75.) The emission scales replace one log s_hat per removed leg by that leg's log q^2; the blend replaces a fraction
alpha of it (and the same with Lq0, every massive particle at s_hat: feat "m0"):
    blend(alpha) = y + (1 - alpha) (N - 4) log s_hat + alpha Lq,     alpha = 0 the DlessInOrd target, 1 the DlessInOrdQ one
both with the coupling orders divided out. Exact per process from the means and covariances of norm_targets_stats.json
(the blend is linear in them). Read with norm_targets.py's three proxies (range, the composition level model, the
within-process spread); the level model on the composition alone and with the pool's label-free kinematic summary
added (its mean of Lq - (N - 4) log s_hat, computable from momenta for any process); first checked against the GPU result: the proxy must rank alpha = 0 and 1 as the A/B did on
both sets before a blend is worth a GPU test.
    .venv-driver/bin/python analysis/catalog_daniel/norm_blend.py
Output: the table printed, data/norm_blend.json.
"""
import json, os, sys

import numpy as np
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from norm_targets import A_EM, A_S, comp  # noqa: E402

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def blend(r, alpha, ord_, col=6):
    """col 6: Lq (the emission scales); col 7: Lq0 (the same with every massive particle at s_hat)"""
    m, C, N = np.array(r["mean"]), np.array(r["cov"]), r["N"]
    a = np.array([1, (1 - alpha) * (N - 4), 0, 0, 0, 0, 0, 0], float); a[col] = alpha
    q, e = ord_
    return float(a @ m) - q * np.log(4 * np.pi * A_S) - e * np.log(4 * np.pi * A_EM), float(a @ C @ a)


def main():
    st = json.load(open(os.path.join(HERE, "data", "norm_targets_stats.json")))
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(HERE, "classes_default.json")))["processes"]}
    twins = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))

    def orders(s, n):
        if s in ("train", "heldout"):
            o = meta[n]["orders"][0]; return o.get("QCD", 0), o.get("QED", 0)
        par = "dc_" + n.split("_", 1)[1].rsplit("_", 1)[0]
        if par not in meta:
            return None
        o = meta[par]["orders"][0]; add = n.rsplit("_", 1)[1]
        return o.get("QCD", 0) + (add == "g"), o.get("QED", 0) + (add == "a")

    rows = {(s, n): r for s in ("train", "heldout", "2to4", "2to4c30") for n, r in st[s].items()
            if not r.get("signed") and orders(s, n) is not None}
    tr = [k for k in rows if k[0] == "train"]
    sets = {"43 new": [k for k in rows if k[0] == "heldout" and k[1] not in twins],
            "43 new, 2->3": [k for k in rows if k[0] == "heldout" and k[1] not in twins and rows[k]["N"] == 5],
            "2to4 +g": [k for k in rows if k[0] == "2to4" and k[1].endswith("_g")],
            "2to4 +a": [k for k in rows if k[0] == "2to4" and k[1].endswith("_a")]}
    # the label-free kinematic summary of a pool: its mean of Lq - (N - 4) log s_hat (momenta only, no amplitude, so
    # available for a process never trained on), as an extra input of the level model (feat = "comp+kin")
    kin = lambda k: float(rows[k]["mean"][6] - (rows[k]["N"] - 4) * rows[k]["mean"][1])
    res = {}
    for al, feat in ([(x, "comp") for x in ALPHAS] + [(x, "comp+kin") for x in (0.0,)]
                     + [(x, "m0") for x in ALPHAS[1:]]):
        X_of = (lambda k: np.r_[comp(rows[k]["pdg"]), kin(k)]) if feat == "comp+kin" else (lambda k: comp(rows[k]["pdg"]))
        Xtr = np.array([X_of(k) for k in tr])
        b = {k: blend(rows[k], al, orders(*k), 7 if feat == "m0" else 6) for k in rows}
        ytr = np.array([b[k][0] for k in tr])
        lo, med, hi = ytr.min(), np.median(ytr), ytr.max()
        iqr = np.subtract(*np.quantile(ytr, [0.75, 0.25]))
        mdl = Ridge(alpha=1e-2).fit(Xtr, ytr)
        r = {"train": [float(lo), float(med), float(hi)], "shape_sd_train": float(np.median([np.sqrt(b[k][1]) for k in tr]))}
        for s, ks in sets.items():
            y = np.array([b[k][0] for k in ks]); X = np.array([X_of(k) for k in ks])
            r[s] = {"n": len(ks), "above_max": float(np.mean(y > hi)), "dist_iqr": float(np.median((y - med) / iqr)),
                    "level": float(np.median((mdl.predict(X) - y) ** 2)),
                    "shape_sd": float(np.median([np.sqrt(b[k][1]) for k in ks]))}
        res[f"{al}|{feat}"] = r
        print(f"{feat:8s} alpha {al:4.2f} sd {r['shape_sd_train']:.2f} | " + " | ".join(
            f"{s} (n={r[s]['n']}): above {r[s]['above_max']:.2f} level {r[s]['level']:.3g} sd {r[s]['shape_sd']:.2f}"
            for s in sets))
    json.dump(res, open(os.path.join(HERE, "data", "norm_blend.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
