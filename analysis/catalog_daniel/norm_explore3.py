"""D83 exploration, third pass: the small network that places a process's level, with the diagrams as extra
descriptors and on a sampling-robust level. Targets of norm_explore2.py: B (one shared standardization) and A + IDEN +
colour. Process descriptors: stats_net.py's features, optionally with the diagrams' (diagram_couplings.json): the
leading coupling product, log of the sum over diagrams minus the leading one, log of the number of diagrams, the
leading diagram's massive internal lines and the most any diagram has. Level: as sampled (the pool's mean) or
reweighted flat in log s_hat (the mean over the 20 bins of norm_moments2.py, each bin one weight), the second less
tied to how the pool drew s_hat. Network: sklearn MLPRegressor 2x64, 3 seeds averaged, cross-fitted by relabelling
class on the training processes; scored on the 43 new held-out processes by the level error against the same level
definition. Output data/norm_explore3.json.
    python analysis/catalog_daniel/norm_explore3.py
"""
import json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from norm_explore import iden, signed_pdgs, COLOR  # noqa: E402
import stats_net as sn  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def main():
    mo = json.load(open(os.path.join(HERE, "data", "norm_moments2.json")))
    dc = json.load(open(os.path.join(HERE, "data", "diagram_couplings.json")))
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(HERE, "classes_default.json")))["processes"]}
    rel = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    tr, ho = sorted(mo["train"]), sorted(mo["heldout"])
    R = {**mo["train"], **mo["heldout"]}
    new = [n for n in ho if n not in rel]
    grp = np.array([meta[n]["weak_key"] for n in tr])
    ls0 = math.log(289.0)

    def offset(n):
        ins, outs = signed_pdgs(meta[n])
        return max(dc[n]["logc2"]) - math.log(iden(ins, outs)) + sum(math.log(COLOR.get(abs(p), 1)) for p in outs)

    def level(n, target, flat):
        r = R[n]
        w = np.zeros(len(mo["vars"])); w[0] = 1.0
        o = 0.0
        if target == "A+":
            w[1] = -(4 - r["N"]); o = offset(n) - (4 - r["N"]) * ls0
        sd = float(np.sqrt(w @ np.array(r["cov"]) @ w))
        if flat:
            bm, bc = np.array(r["bin_mean"]), np.array(r["bin_count"])
            return float((bm[bc > 0] @ w).mean()) - o, sd
        return float(np.dot(r["mean"], w)) - o, sd

    def feats(n, diag):
        f = list(sn.features(meta[n]))
        if diag:
            l = np.array(dc[n]["logc2"]); mi = dc[n]["massive_internal"]
            f += [float(l.max()), float(np.log(np.exp(l - l.max()).sum())), float(np.log(len(l))),
                  float(mi[int(np.argmax(l))]), float(max(mi))]
        return np.array(f, float)

    def mlp(r):
        return MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-2, learning_rate_init=3e-3, max_iter=3000,
                            early_stopping=True, validation_fraction=0.15, random_state=r)

    out = {}
    for target in ("B", "A+"):
        for diag in (False, True):
            for flat in (False, True):
                L = {n: level(n, target, flat) for n in R}
                M = np.mean([L[n][0] for n in tr])
                Xtr = np.array([feats(n, diag) for n in tr]); ytr = np.array([L[n][0] - M for n in tr])
                sd = np.array([L[n][1] for n in tr])
                cf = np.zeros(len(tr))
                for i, j in GroupKFold(5).split(Xtr, ytr, grp):
                    s2 = StandardScaler().fit(Xtr[i])
                    cf[j] = np.mean([mlp(r).fit(s2.transform(Xtr[i]), ytr[i]).predict(s2.transform(Xtr[j]))
                                     for r in range(3)], 0)
                sc = StandardScaler().fit(Xtr)
                nets = [mlp(r).fit(sc.transform(Xtr), ytr) for r in range(3)]
                lv = {}
                for n in ho:
                    p = np.mean([m_.predict(sc.transform(feats(n, diag)[None]))[0] for m_ in nets])
                    lv[n] = (p - (L[n][0] - M)) ** 2 / L[n][1] ** 2
                cfl = (cf - ytr) ** 2 / sd ** 2
                key = f"{target}|{'diag' if diag else 'nodiag'}|{'flat' if flat else 'sampled'}"
                out[key] = dict(crossfit=float(np.median(cfl)), crossfit90=float(np.quantile(cfl, .9)),
                                new=float(np.median([lv[n] for n in new])), new90=float(np.quantile([lv[n] for n in new], .9)),
                                rel=float(np.median([lv[n] for n in ho if n in rel])),
                                worst=sorted(((meta[n]["process"], lv[n]) for n in new), key=lambda t: -t[1])[:5])
                v = out[key]
                print(f"{key:24s} cross-fit {v['crossfit']:.3g} (90% {v['crossfit90']:.3g}) | 43 new {v['new']:.3g} "
                      f"(90% {v['new90']:.3g}) | 96 rel {v['rel']:.3g} | worst " +
                      "; ".join(f"{p} {e:.3g}" for p, e in v["worst"][:3]), flush=True)
    json.dump(out, open(os.path.join(HERE, "data", "norm_explore3.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
