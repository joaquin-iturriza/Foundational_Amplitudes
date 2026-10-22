"""P1 follow-ups on norm_surrogate.py's statistics. (a) The surrogate as a per-event target factor: y - beta s with one
beta per multiplicity fitted on the training processes (within-process covariance), and the within-process std of the
residual over that of y (median over the 43 new and 96 relabels; 1 = no gain). (b) A g-like level network (2x64 MLP,
3 seeds, as norm_explore3.py) on the process features + diagram descriptors, with and without the surrogates'
per-process mean, log std and correlation-free statistics (no labels) as extra inputs; level error on the 43 new and
96 relabels, cross-fitted by multiplicity too.
    .venv-driver/bin/python analysis/catalog_daniel/norm_surrogate_more.py
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
import stats_net as sn  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

S = {k: v for k, v in json.load(open(os.path.join(HERE, "data", "surrogate.json"))).items() if "error" not in v}
rel = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
held = set(json.load(open(os.path.join(HERE, "data", "norm_moments2.json")))["heldout"])
meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(HERE, "classes_default.json")))["processes"]}
dc = json.load(open(os.path.join(HERE, "data", "diagram_couplings.json")))
tr = sorted(n for n in S if n not in held); ho = sorted(n for n in S if n in held)
new = [n for n in ho if n not in rel]; rl = [n for n in ho if n in rel]
out = {}
print("(a) residual within-process std / std of y, beta per multiplicity:")
for k in ("dim", "eik", "pt", "prop", "propeik"):
    beta = {}
    for N in (4, 5):
        ns = [n for n in tr if S[n]["N"] == N]
        cov = sum(S[n][k][2] * S[n][k][1] * S[n]["y_std"] for n in ns); var = sum(S[n][k][1] ** 2 for n in ns)
        beta[N] = cov / var if var > 0 else 0.0
    def ratio(n):
        b, sy, ss, c = beta[S[n]["N"]], S[n]["y_std"], S[n][k][1], S[n][k][2]
        return float(np.sqrt(max(sy ** 2 - 2 * b * c * sy * ss + b ** 2 * ss ** 2, 0)) / sy)
    r = {lab: float(np.median([ratio(n) for n in names])) for lab, names in (("train", tr), ("new", new), ("rel", rl))}
    out[f"factor|{k}"] = dict(beta=beta, **r)
    print(f"  {k:8s} beta {beta[4]:.2f}, {beta[5]:.2f}: train {r['train']:.2f}, 43 new {r['new']:.2f}, 96 rel {r['rel']:.2f}")


def pf(n, sur):
    f = list(sn.features(meta[n]))
    l = np.array(dc[n]["logc2"]); mi = dc[n]["massive_internal"]
    f += [float(l.max()), float(np.log(np.exp(l - l.max()).sum())), float(np.log(len(l))), float(mi[int(np.argmax(l))]),
          float(max(mi))]
    if sur:
        for k in ("dimid", "eik", "pt", "prop", "propeik"):
            f += [S[n][k][0], float(np.log(max(S[n][k][1], 1e-6)))]
    return np.array(f, float)


def mlp(r):
    return MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-2, learning_rate_init=3e-3, max_iter=3000,
                        early_stopping=True, validation_fraction=0.15, random_state=r)


print("(b) level network, level error (median, 90%):")
for sur in (False, True):
    for fit_on, test, lab in ((tr, {"new": new, "rel": rl}, "held-out"),
                              ([n for n in tr if S[n]["N"] == 4], {"2to3": [n for n in S if S[n]["N"] == 5]}, "2->2 to 2->3")):
        X = np.array([pf(n, sur) for n in fit_on]); M = np.mean([S[n]["y_mean"] for n in fit_on])
        y = np.array([S[n]["y_mean"] - M for n in fit_on])
        sc = StandardScaler().fit(X); nets = [mlp(r).fit(sc.transform(X), y) for r in range(3)]
        res = {}
        for t, names in test.items():
            Xt = sc.transform(np.array([pf(n, sur) for n in names]))
            p = np.mean([m.predict(Xt) for m in nets], 0)
            e = (p + M - np.array([S[n]["y_mean"] for n in names])) ** 2 / np.array([S[n]["y_std"] for n in names]) ** 2
            res[t] = (float(np.median(e)), float(np.quantile(e, .9)))
        out[f"net|{'sur' if sur else 'nosur'}|{lab}"] = res
        print(f"  {'with' if sur else 'without'} surrogates, fit {lab}: " + ", ".join(f"{t} {v[0]:.3g} (90% {v[1]:.3g})" for t, v in res.items()))
json.dump(out, open(os.path.join(HERE, "data", "surrogate_more.json"), "w"), indent=1)
