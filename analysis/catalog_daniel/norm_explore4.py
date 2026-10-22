"""D83 exploration, fourth pass: how robust is norm_explore3.py's best result (B's level of the 43 new held-out processes
placed by a small network on the process features plus the diagram descriptors, median level error 0.045)?
  - seeds: 10 seeds of the 3-seed ensemble (each a different split of the early-stopping set)
  - size: hidden (32,32), (64,64), (128,128), (64,64,64); weight decay alpha 1e-3, 1e-2, 1e-1
  - cross-fit: by relabelling class (as before) and leave-one-multiplicity-out (the network never saw a process with
    that many particles: a harsher extrapolation than the held-out split asks for)
  - few-shot reference: the level read off the first k events of each held-out process (k = 10, 100), as if a few
    events of a new process were known; its error is the sampling error of a k-event mean
Level error: (predicted level - level)^2 / within-process variance, median over the 43 new (and the 96 relabels).
Target B: one shared standardization, level = the pool's mean log|M|^2 (norm_moments2.json, 1e5 events per process).
Output data/norm_explore4.json.
    .venv-driver/bin/python analysis/catalog_daniel/norm_explore4.py
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import stats_net as sn  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def main():
    mo = json.load(open(os.path.join(HERE, "data", "norm_moments2.json")))
    dc = json.load(open(os.path.join(HERE, "data", "diagram_couplings.json")))
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(HERE, "classes_default.json")))["processes"]}
    rel = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    hp = dict(np.load(os.path.join(HERE, "data", "heldout_preds.npz")))
    tr, ho = sorted(mo["train"]), sorted(mo["heldout"])
    R = {**mo["train"], **mo["heldout"]}
    new = [n for n in ho if n not in rel]
    lev = {n: R[n]["mean"][0] for n in R}
    sd = {n: float(np.sqrt(R[n]["cov"][0][0])) for n in R}
    M = np.mean([lev[n] for n in tr])

    def feats(n):
        f = list(sn.features(meta[n]))
        l = np.array(dc[n]["logc2"]); mi = dc[n]["massive_internal"]
        f += [float(l.max()), float(np.log(np.exp(l - l.max()).sum())), float(np.log(len(l))),
              float(mi[int(np.argmax(l))]), float(max(mi))]
        return np.array(f, float)

    X = {n: feats(n) for n in R}
    Xtr = np.array([X[n] for n in tr]); ytr = np.array([lev[n] - M for n in tr]); sdt = np.array([sd[n] for n in tr])
    Xho = np.array([X[n] for n in ho])

    def fit_predict(Xa, ya, Xb, hidden, alpha, seeds):
        sc = StandardScaler().fit(Xa)
        return np.mean([MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=3e-3, max_iter=3000,
                                     early_stopping=True, validation_fraction=0.15, random_state=r)
                        .fit(sc.transform(Xa), ya).predict(sc.transform(Xb)) for r in seeds], 0)

    def score(pred):
        e = {n: (p - (lev[n] - M)) ** 2 / sd[n] ** 2 for n, p in zip(ho, pred)}
        return (float(np.median([e[n] for n in new])), float(np.quantile([e[n] for n in new], .9)),
                float(np.median([e[n] for n in ho if n in rel])))

    out = {"seeds": [], "size": {}, "crossfit": {}, "fewshot": {}}
    for s0 in range(10):
        out["seeds"].append(score(fit_predict(Xtr, ytr, Xho, (64, 64), 1e-2, range(3 * s0, 3 * s0 + 3))))
    a = np.array(out["seeds"])
    print(f"10 ensembles of 3 seeds, (64,64), alpha 1e-2: 43 new median {np.median(a[:, 0]):.3g} "
          f"(range {a[:, 0].min():.3g} to {a[:, 0].max():.3g}); 90% {np.median(a[:, 1]):.3g}; relabels {np.median(a[:, 2]):.3g}",
          flush=True)
    for hidden in ((32, 32), (64, 64), (128, 128), (64, 64, 64)):
        for alpha in (1e-3, 1e-2, 1e-1):
            v = score(fit_predict(Xtr, ytr, Xho, hidden, alpha, range(3)))
            out["size"][f"{hidden}|{alpha}"] = v
            print(f"  {str(hidden):14s} alpha {alpha:g}: 43 new {v[0]:.3g} (90% {v[1]:.3g}), relabels {v[2]:.3g}", flush=True)
    # cross-fit on the training processes
    grp_cls = np.array([meta[n]["weak_key"] for n in tr])
    grp_N = np.array([R[n]["N"] for n in tr])
    for name, grp, k in (("class", grp_cls, 5), ("multiplicity", grp_N, len(set(grp_N)))):
        cf = np.zeros(len(tr))
        for i, j in GroupKFold(k).split(Xtr, ytr, grp):
            cf[j] = fit_predict(Xtr[i], ytr[i], Xtr[j], (64, 64), 1e-2, range(3))
        e = (cf - ytr) ** 2 / sdt ** 2
        per_N = {int(N): float(np.median(e[grp_N == N])) for N in sorted(set(grp_N))}
        out["crossfit"][name] = dict(median=float(np.median(e)), p90=float(np.quantile(e, .9)), per_N=per_N)
        print(f"cross-fit by {name}: median {np.median(e):.3g} (90% {np.quantile(e, .9):.3g}); per N {per_N}", flush=True)
    # few-shot: the level from the first k validation events of each held-out process (heldout_preds.npz truth)
    for k in (10, 100):
        e = {}
        for n in ho:
            t = hp[f"{n}__t"].astype(float)
            e[n] = (t[:k].mean() - lev[n]) ** 2 / sd[n] ** 2
        out["fewshot"][k] = (float(np.median([e[n] for n in new])), float(np.median([e[n] for n in ho if n in rel])))
        print(f"few-shot level from {k} events: 43 new {out['fewshot'][k][0]:.3g}, relabels {out['fewshot'][k][1]:.3g}")
    json.dump(out, open(os.path.join(HERE, "data", "norm_explore4.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
