"""D83 exploration, seventh pass: the few-shot affine of norm_explore5.py with the slope shrunk toward 1 (the level
free, a penalty lam on (b - 1)^2 per unit of the k events' spread in q), for A and B on k = 3, 10 random events (5
draws). MSE(log|M|^2) on the other events, median and 90% over the 43 new and the 96 relabels. Output
data/norm_explore7.json.
    python analysis/catalog_daniel/norm_explore7.py
"""
import json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
tw = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
rng = np.random.default_rng(0)
out = {}
for arm in "AB":
    z = dict(np.load(os.path.join(HERE, "data", f"catalog_norm_{arm}.npz")))
    P = sorted({k.split("__")[0] for k in z if k.endswith("__t")})
    for k in (3, 10):
        for lam in (0.0, 1.0, 10.0, 100.0):
            res = {False: [], True: []}
            for p in P:
                q, t, e = z[p + "__q"].astype(float), z[p + "__t"].astype(float), []
                for _ in range(5):
                    idx = rng.choice(len(t), k, replace=False); rest = np.setdiff1d(np.arange(len(t)), idx)
                    qc = q[idx] - q[idx].mean()
                    b = 1 + (qc @ (t[idx] - q[idx])) / (qc @ qc + lam * max(qc @ qc / k, 1e-9))
                    a = np.mean(t[idx] - b * q[idx])
                    e.append(np.mean((a + b * q[rest] - t[rest]) ** 2))
                res[p in tw].append(float(np.median(e)))
            key = f"{arm}|k={k}|lam={lam:g}"
            out[key] = dict(new=float(np.median(res[False])), new90=float(np.quantile(res[False], .9)),
                            rel=float(np.median(res[True])))
            print(key, out[key])
json.dump(out, open(os.path.join(HERE, "data", "norm_explore7.json"), "w"), indent=1)
