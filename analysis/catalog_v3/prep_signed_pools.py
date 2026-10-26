"""The signed-data check on catalog_v3 (the preprocessing exploration, D83/D114; the part the shape proxy leaves apart,
prep_shape.py): on every one-loop pool of catalog_v3's training recipe and of the one-loop held-out axis (H5), how
often the stored value is negative, how much of the sign the event's kinematics decide, and the shape a fixed small
regressor leaves on the magnitude under each candidate target.
Per pool, on its validation and test events concatenated (10^4 events, the roles every pool has; a held-out pool has
no train split), the first n_fit to fit and the next n_score to score:
    neg            share of negative events (all 10^4); a pool with none, or all, is listed as not sign-changing
    near10/100     share within a factor 10 / 100 of zero: |x| below the median of max(|x|, floor) over 10 / 100
                   (floor: the 1% quantile of |x| on the fit events, the sign head's abslog floor, data.signedlog_quantile)
    err_major      wrong-sign share on the scoring events of one sign per pool (the fit events' majority)
    err_mlp        the same for a classifier on the event's invariants (sklearn MLPClassifier (64, 64), early stopping,
                   seed 0; the inputs of prep_shape.py: log(2|p_i.p_j|/s_hat) over pairs, log(m^2/s_hat) of the
                   massive legs, log s_hat; standardized on the fit events), and err_mlp_near on the scoring events
                   within a factor 10 of zero
    err_shat       the same classifier on log s_hat alone (does the sign follow the energy only)
    shape_<t>      the regressor of prep_shape.py (MLPRegressor (64, 64)) on the magnitude log max(|x|, floor) under
                   target t: raw (as is), dless (+ (N - 4) log s_hat), qperp (+ the emission scales Lq); its scoring
                   MSE after the mean, and that over the target's variance (frac_<t>)
    python analysis/catalog_v3/prep_signed_pools.py [--workers 32] [--n-fit 7500 --n-score 2500]
                                                     (a CPU job on CC: scripts/job_prep_signed_v3.sh)
Output: one SIGNED_DATA JSON line per pool (an error or a missing pool listed, not dropped); the read is
prep_signed.py on these lines copied home into data/prep_signed_raw.txt.
"""
import argparse, glob, json, os, sys
from multiprocessing import Pool

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from physics_scales import minkowski_dots, qperp_logs  # noqa: E402

TARGETS = ("raw", "dless", "qperp")
FLOOR_Q = 0.01                       # the sign head's abslog floor quantile (data.signedlog_quantile default)


def files(name, sq):
    d = os.environ["DATA_DIR"]
    out = []
    for role in ("val", "test"):
        g = sorted(glob.glob(os.path.join(d, f"{name}_{sq[0]}-{sq[1]}GeV_{role}_*amplitudes.npy"))) if sq else []
        g = g or sorted(glob.glob(os.path.join(d, f"{name}_*_{role}_*amplitudes.npy")))
        g = [f for f in g if not f.endswith(".alphas.npy")]
        if g:
            out.append(g[0])
    return out


def one(args):
    setname, name, sq, n_fit, n_score = args
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    fs = files(name, sq)
    if not fs:
        return setname, name, {"error": "no pool"}
    x = np.concatenate([np.asarray(np.load(f, mmap_mode="r"), dtype=np.float64) for f in fs])
    N = (x.shape[1] - 1) // 5
    P = x[:, : 4 * N].reshape(len(x), N, 4); amp = x[:, -1]
    rec = {"N": int(N), "pdg": [int(c) for c in x[0, 4 * N: 5 * N]], "n": int(len(x)), "files": [os.path.basename(f) for f in fs]}
    if not np.all(np.isfinite(amp)):
        return setname, name, {**rec, "error": "non-finite"}
    neg = float(np.mean(amp < 0))
    rec["neg"] = neg
    if neg in (0.0, 1.0):
        rec["signed"] = False
        return setname, name, rec
    rec["signed"] = True
    if len(x) < n_fit + n_score:
        return setname, name, {**rec, "error": f"{len(x)} events < {n_fit + n_score}"}
    x, P, amp = x[: n_fit + n_score], P[: n_fit + n_score], amp[: n_fit + n_score]
    a = np.abs(amp)
    floor = float(np.quantile(a[:n_fit][a[:n_fit] > 0], FLOOR_Q))
    m = np.log(np.maximum(a, floor))
    med = np.median(np.maximum(a, floor))
    rec.update({"floor": floor, "near10": float(np.mean(a < med / 10)), "near100": float(np.mean(a < med / 100))})
    G = minkowski_dots(P)
    Ls = np.log(np.abs(G[:, 0, 0] + G[:, 1, 1] + 2 * G[:, 0, 1]))
    iu = np.triu_indices(N, 1)
    X = [np.log(np.clip(2 * np.abs(G[:, iu[0], iu[1]]), 1e-300, None)) - Ls[:, None]]
    m2 = np.einsum("nii->ni", G)
    massive = np.where(np.median(m2, 0) > 1.0)[0]
    if len(massive):
        X.append(np.log(np.clip(m2[:, massive], 1e-300, None)) - Ls[:, None])
    X.append(Ls[:, None])
    X = np.concatenate(X, 1)
    X = (X - X[:n_fit].mean(0)) / (X[:n_fit].std(0) + 1e-12)
    s = amp > 0
    sf, ss = s[:n_fit], s[n_fit:]
    near = a[n_fit:] < med / 10
    major = bool(sf.mean() >= 0.5)
    rec["neg_score"] = float(np.mean(~ss))
    rec["err_major"] = float(np.mean(ss != major))
    rec["err_major_near"] = float(np.mean(ss[near] != major)) if near.any() else None
    c = MLPClassifier(hidden_layer_sizes=(64, 64), learning_rate_init=1e-3, max_iter=300, early_stopping=True,
                      random_state=0).fit(X[:n_fit], sf)
    p = c.predict(X[n_fit:])
    rec["err_mlp"] = float(np.mean(p != ss))
    rec["err_mlp_near"] = float(np.mean(p[near] != ss[near])) if near.any() else None
    Z = ((Ls - Ls[:n_fit].mean()) / (Ls[:n_fit].std() + 1e-12))[:, None]
    c1 = MLPClassifier(hidden_layer_sizes=(16,), learning_rate_init=1e-3, max_iter=300, early_stopping=True,
                       random_state=0).fit(Z[:n_fit], sf)
    rec["err_shat"] = float(np.mean(c1.predict(Z[n_fit:]) != ss))
    tg = {"raw": m, "dless": m + (N - 4) * Ls, "qperp": m + qperp_logs(P)}
    for k, y in tg.items():
        mu, sd = y[:n_fit].mean(), y[:n_fit].std()
        r = MLPRegressor(hidden_layer_sizes=(64, 64), learning_rate_init=1e-3, max_iter=300, early_stopping=True,
                         random_state=0).fit(X[:n_fit], (y[:n_fit] - mu) / sd)
        e = (r.predict(X[n_fit:]) * sd + mu) - y[n_fit:]
        rec[f"shape_{k}"] = float(np.mean((e - e.mean()) ** 2))
        rec[f"frac_{k}"] = float(np.mean((e - e.mean()) ** 2) / np.var(y[n_fit:]))
    return setname, name, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--n-fit", type=int, default=7500)
    ap.add_argument("--n-score", type=int, default=2500)
    a = ap.parse_args()
    jobs = []
    for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v3_train.yaml")))["processes"]:
        if p["name"].startswith("v3_") and ("_nlo" in p["name"] or "_loop" in p["name"]):
            jobs.append(("train", p["name"], p.get("sqrts"), a.n_fit, a.n_score))
    for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v3_heldout_H5.yaml")))["processes"]:
        jobs.append(("H5", p["name"], p.get("sqrts"), a.n_fit, a.n_score))
    print("JOBS", len(jobs), flush=True)
    with Pool(a.workers) as pl:
        for s, n, r in pl.imap_unordered(one, jobs, chunksize=1):
            print("SIGNED_DATA " + json.dumps({"set": s, "name": n, **r}), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
