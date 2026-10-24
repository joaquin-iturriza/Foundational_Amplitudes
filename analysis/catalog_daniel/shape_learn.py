"""D83: why does the network fit the emission-scales target's shape worse on the 43 new processes (par:qperp_cost: the
non-boson finals' shape 0.28 -> 0.63) although its within-process spread is narrower? The composition proxy reads only
the level; this is a proxy for the shape: per process, one small regressor on the event's invariants, the same for every
target, and its held-out error. A target the regressor fits worse is one a network has to work harder for.
Targets per event (orders and constants drop out of a per-process shape):
    dless   log|M|^2 + (N - 4) log s_hat
    qperp   log|M|^2 + Lq                            (physics_scales.qperp_logs, the DlessInOrdQ target)
    soft    log|M|^2 + Lq with tau = 0.3             (the soft minimum: no kink where the minimising pair changes)
    soft1   the same, tau = 0.1
Inputs: log(2|p_i.p_j| / s_hat) over every pair, log(m_i^2 / s_hat) of the massive ones; standardized. Regressor:
sklearn MLPRegressor (64, 64), Adam, early stopping, seed 0; 15000 events to fit, 5000 to score; the score is the
held-out MSE over the target's variance (1 - R^2) and the MSE itself. Sets: the 43 new processes (train pools) and the
2->4 +g probes (val pools), as in norm_targets_stats.py.
    python analysis/catalog_daniel/shape_learn.py [--workers 16]   (a CPU job on CC: scripts/job_shape_learn.sh)
Output: $SCRATCH/zz_shape_learn/shape_learn.json and one SHAPE line per process.
"""
import argparse, json, os, sys
from multiprocessing import Pool

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from physics_scales import minkowski_dots, qperp_logs  # noqa: E402
from norm_targets_stats import SETS, pool  # noqa: E402

N_FIT, N_SCORE = 15000, 5000


def one(args):
    setname, name, sq, role = args
    from sklearn.neural_network import MLPRegressor
    f = pool(name, sq, role)
    if f is None:
        return setname, name, None
    x = np.asarray(np.load(f, mmap_mode="r")[: N_FIT + N_SCORE], dtype=np.float64)
    N = (x.shape[1] - 1) // 5
    P = x[:, : 4 * N].reshape(len(x), N, 4); amp = x[:, -1]
    if np.any(amp <= 0) or len(x) < N_FIT + N_SCORE:
        return setname, name, None
    G = minkowski_dots(P)
    Ls = np.log(np.abs(G[:, 0, 0] + G[:, 1, 1] + 2 * G[:, 0, 1]))
    iu = np.triu_indices(N, 1)
    X = [np.log(np.clip(2 * np.abs(G[:, iu[0], iu[1]]), 1e-300, None)) - Ls[:, None]]
    m2 = np.einsum("nii->ni", G)
    massive = np.where(np.median(m2, 0) > 1.0)[0]
    if len(massive):
        X.append(np.log(np.clip(m2[:, massive], 1e-300, None)) - Ls[:, None])
    X = np.concatenate(X, 1)
    X = (X - X[:N_FIT].mean(0)) / (X[:N_FIT].std(0) + 1e-12)
    y0 = np.log(amp)
    tg = {"dless": y0 + (N - 4) * Ls, "qperp": y0 + qperp_logs(P), "soft": y0 + qperp_logs(P, soft=0.3),
          "soft1": y0 + qperp_logs(P, soft=0.1)}
    out = {"N": int(N), "pdg": [int(c) for c in x[0, 4 * N: 5 * N]]}
    for k, y in tg.items():
        mu, sd = y[:N_FIT].mean(), y[:N_FIT].std()
        r = MLPRegressor((64, 64), learning_rate_init=1e-3, max_iter=300, early_stopping=True, random_state=0)
        r.fit(X[:N_FIT], (y[:N_FIT] - mu) / sd)
        e = (r.predict(X[N_FIT:]) * sd + mu) - y[N_FIT:]
        out[k] = {"mse": float(np.mean((e - e.mean()) ** 2)), "var": float(np.var(y[N_FIT:])),
                  "frac": float(np.mean((e - e.mean()) ** 2) / np.var(y[N_FIT:]))}
    return setname, name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    tw = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    jobs = []
    for s in ("heldout", "2to4"):
        rec, role = SETS[s]
        for p in yaml.safe_load(open(rec))["processes"]:
            if s == "heldout" and p["name"] in tw:
                continue
            if s == "2to4" and not p["name"].endswith("_g"):
                continue
            jobs.append((s, p["name"], p.get("sqrts"), role))
    res = {}
    with Pool(a.workers) as pl:
        for s, n, r in pl.imap_unordered(one, jobs):
            if r is None:
                continue
            res.setdefault(s, {})[n] = r
            print("SHAPE", s, n, json.dumps({k: round(v["mse"], 4) for k, v in r.items() if isinstance(v, dict)}), flush=True)
    od = os.path.join(os.environ.get("SCRATCH", "/tmp"), "zz_shape_learn"); os.makedirs(od, exist_ok=True)
    json.dump(res, open(os.path.join(od, "shape_learn.json"), "w"))
    for s, d in res.items():
        print("SUMMARY", s, len(d), {k: round(float(np.median([r[k]["mse"] for r in d.values()])), 4)
                                     for k in ("dless", "qperp", "soft", "soft1")})


if __name__ == "__main__":
    main()
