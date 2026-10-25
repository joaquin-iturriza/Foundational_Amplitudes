"""The shape proxy on catalog_v3 (the preprocessing exploration, D83/D99, moved off Daniel's catalog by D114): per class,
one small regressor on the event's invariants, the same for every target, and its held-out error. The zero-shot error
left on catalog_v3 is shape, not level (par. The level's two targets on catalog_v3: 0.79 to 0.94 on 2->4, 0.80 to 0.88
on the top); a target the regressor fits worse within a process is one a network has to work harder for. The port of
analysis/catalog_daniel/shape_learn.py.
Targets per event (orders and constants drop out of a per-process shape):
    raw     log|M|^2                                   (the baseline arm's target)
    dless   log|M|^2 + (N - 4) log s_hat               (the s_hat factor alone)
    qperp   log|M|^2 + Lq                              (physics_scales.qperp_logs: the emission scales, the
                                                         dimensionless arm's target_physics_dim_ref=qperp)
    soft    log|M|^2 + Lq with tau = 0.3               (the soft minimum: no kink where the minimising pair changes)
Inputs, the same for every target: log(2|p_i.p_j| / s_hat) over every pair, log(m_i^2 / s_hat) of the massive ones, and
log s_hat itself (catalog_v3's windows span 25 GeV to 1 TeV, so a dimensionful target varies with it); standardized.
Regressor: sklearn MLPRegressor (64, 64), Adam, early stopping, seed 0; on each class's validation pool (10^4 events,
the size every role shares) 7500 events to fit, 2500 to score; the score is the held-out MSE of the target (after its
mean) and that MSE over the target's variance. Sets: 200 training classes drawn with seed 0 from catalog_v3_train.yaml's
tree classes, and every class of the held-out axes H1, H2, H2p5, H3, H6, H7c, H7E (H5's one-loop pools change sign:
the signed check is apart). Pools with a non-positive |M|^2 or fewer events are listed as skipped, with the reason.
    python analysis/catalog_v3/prep_shape.py [--workers 32] [--n-fit 7500 --n-score 2500] [--tag ""]
                                              (a CPU job on CC: scripts/job_prep_shape_v3.sh)
Output: $SCRATCH/zz_v3_prep/prep_shape<tag>.json and one SHAPE line per class.
"""
import argparse, glob, json, os, sys
from multiprocessing import Pool

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from physics_scales import minkowski_dots, qperp_logs  # noqa: E402

AXES = ("H1", "H2", "H2p5", "H3", "H6", "H7c", "H7E")
TARGETS = ("raw", "dless", "qperp", "soft")


def pool(name, sq):
    d = os.environ["DATA_DIR"]
    g = sorted(glob.glob(os.path.join(d, f"{name}_{sq[0]}-{sq[1]}GeV_val_*amplitudes.npy"))) if sq else []
    g = g or sorted(glob.glob(os.path.join(d, f"{name}_*_val_*amplitudes.npy")))
    return g[0] if g else None


def one(args):
    setname, name, sq, n_fit, n_score = args
    from sklearn.neural_network import MLPRegressor
    f = pool(name, sq)
    if f is None:
        return setname, name, "no pool"
    x = np.asarray(np.load(f, mmap_mode="r")[: n_fit + n_score], dtype=np.float64)
    if len(x) < n_fit + n_score:
        return setname, name, f"{len(x)} events < {n_fit + n_score}"
    N = (x.shape[1] - 1) // 5
    P = x[:, : 4 * N].reshape(len(x), N, 4); amp = x[:, -1]
    if np.any(amp <= 0) or not np.all(np.isfinite(amp)):
        return setname, name, "signed or non-finite"
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
    y0 = np.log(amp)
    Lq = qperp_logs(P)
    tg = {"raw": y0, "dless": y0 + (N - 4) * Ls, "qperp": y0 + Lq, "soft": y0 + qperp_logs(P, soft=0.3)}
    out = {"N": int(N), "pdg": [int(c) for c in x[0, 4 * N: 5 * N]]}
    for k, y in tg.items():
        mu, sd = y[:n_fit].mean(), y[:n_fit].std()
        r = MLPRegressor(hidden_layer_sizes=(64, 64), learning_rate_init=1e-3, max_iter=300, early_stopping=True,
                         random_state=0)
        r.fit(X[:n_fit], (y[:n_fit] - mu) / sd)
        e = (r.predict(X[n_fit:]) * sd + mu) - y[n_fit:]
        out[k] = {"mse": float(np.mean((e - e.mean()) ** 2)), "var": float(np.var(y[n_fit:])),
                  "frac": float(np.mean((e - e.mean()) ** 2) / np.var(y[n_fit:]))}
    return setname, name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--n-fit", type=int, default=7500)
    ap.add_argument("--n-score", type=int, default=2500)
    ap.add_argument("--n-train-classes", type=int, default=200)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    jobs = []
    tr = [p for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v3_train.yaml")))["processes"]
          if p["name"].startswith("v3_") and "_nlo" not in p["name"] and "_loop" not in p["name"]]
    pick = np.random.default_rng(0).permutation(len(tr))[: a.n_train_classes]
    jobs += [("train", tr[i]["name"], tr[i].get("sqrts"), a.n_fit, a.n_score) for i in sorted(pick)]
    for ax in AXES:
        for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", f"catalog_v3_heldout_{ax}.yaml")))["processes"]:
            jobs.append((ax, p["name"], p.get("sqrts"), a.n_fit, a.n_score))
    res, skipped = {}, []
    with Pool(a.workers) as pl:
        for s, n, r in pl.imap_unordered(one, jobs, chunksize=1):
            if isinstance(r, str):
                skipped.append((s, n, r))
                continue
            res.setdefault(s, {})[n] = r
            print("SHAPE", s, n, json.dumps({k: round(r[k]["mse"], 4) for k in TARGETS}), flush=True)
    od = os.path.join(os.environ.get("SCRATCH", "/tmp"), "zz_v3_prep"); os.makedirs(od, exist_ok=True)
    res["skipped"] = skipped
    json.dump(res, open(os.path.join(od, f"prep_shape{a.tag}.json"), "w"))
    print("SKIPPED", len(skipped), skipped[:20])
    for s, d in ((k, v) for k, v in res.items() if k != "skipped"):
        print("SUMMARY", s, len(d), {k: round(float(np.median([r[k]["mse"] for r in d.values()])), 4) for k in TARGETS})


if __name__ == "__main__":
    main()
