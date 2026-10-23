"""D83, step 2 of the 2026-10-23 plan: where the 2->4 shape error sits. Each saved zero-shot prediction of a 2->4 probe
(<PRED>_2to4/<probe>.npz: t, q, logF in event order) is joined to its event in the probe's val pool by its value
(t - logF is the event's log|M|^2), and the error d = q - t, offset removed, is set against the event's kinematics:
    rel_k   Lk - (N-4) log s_hat: the clustering scales against s_hat (norm_targets_stats.py), how soft or collinear
    min_pt  log of the smallest p_T among the massless coloured particles and photons of the final state
    min_dr  the smallest Delta R between two of them
    log_s   log s_hat
    y       log|M|^2
Per probe: the correlation of d with each, the share of d's variance a linear fit on all five explains, and the mean of d
in the five quintiles of rel_k. Login node or CPU job; prints JSON after a __B64__ line.
    python analysis/catalog_daniel/residual_kinematics.py [--arms DlessIn DlessInOrd] [--set 2to4]
"""
import argparse, base64, glob, gzip, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from norm_targets_stats import CLUSTER, cluster_logs, pt2_y_phi  # noqa: E402


def pool_of(probe):
    g = sorted(glob.glob(os.path.join(os.environ["DATA_DIR"], f"{probe}_*_val_*amplitudes.npy")))
    return g[0] if g else None


def features(x):
    N = (x.shape[1] - 1) // 5
    P = x[:, : 4 * N].reshape(len(x), N, 4); pdg = x[0, 4 * N: 5 * N]
    q = P[:, 0] + P[:, 1]; Ls = np.log(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1))
    Lk = cluster_logs(P, pdg, Ls)
    cl = np.array([abs(int(c)) in CLUSTER for c in pdg[2:]])
    pt2, yy, ph = pt2_y_phi(P[:, 2:])
    mpt = 0.5 * np.log(np.where(cl, pt2, np.inf).min(1))
    if cl.sum() >= 2:
        i = np.where(cl)[0]
        dph = np.abs(ph[:, i, None] - ph[:, None, i]); dph = np.minimum(dph, 2 * np.pi - dph)
        dr = np.sqrt((yy[:, i, None] - yy[:, None, i]) ** 2 + dph ** 2) + np.eye(len(i))[None] * 1e9
        mdr = dr.reshape(len(x), -1).min(1)
    else:
        mdr = np.full(len(x), np.nan)
    return {"rel_k": Lk - (N - 4) * Ls, "min_pt": mpt, "min_dr": mdr, "log_s": Ls, "y": np.log(x[:, -1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["DlessIn", "DlessInOrd"])
    ap.add_argument("--set", default="2to4")
    a = ap.parse_args()
    out = {}
    for arm in a.arms:
        d = os.path.join(os.environ["SCRATCH"], f"zz_catalog_norm_short_{arm}_sigstage_{a.set}")
        if not os.path.isdir(d):
            d = os.path.join(os.environ["SCRATCH"], f"zz_catalog_norm_short_{arm}_{a.set}")
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            probe = os.path.basename(f)[:-4]
            z = np.load(f)
            if not z["q"].size:
                continue
            t, q = z["t"].astype(np.float64), z["q"].astype(np.float64)
            y = t - (z["logF"].astype(np.float64) if z["logF"].size else 0.0)
            pf = pool_of(probe)
            if pf is None:
                out[f"{arm}|{probe}"] = {"error": "no val pool"}; continue
            x = np.asarray(np.load(pf), dtype=np.float64)
            ly = np.log(x[:, -1])
            o = np.argsort(ly); j = np.searchsorted(ly[o], y)
            j = np.clip(j, 1, len(o) - 1); j = np.where(np.abs(ly[o][j - 1] - y) < np.abs(ly[o][j] - y), j - 1, j)
            idx = o[j]; err = np.abs(ly[idx] - y)
            ok = err < 1e-4 * np.maximum(1.0, np.abs(y))
            F = features(x[idx[ok]])
            r = (q - t)[ok]; r = r - r.mean()
            X = np.stack([np.nan_to_num(F[k]) for k in F], 1); X = (X - X.mean(0)) / (X.std(0) + 1e-12)
            A = np.concatenate([np.ones((len(X), 1)), X], 1)
            co = np.linalg.lstsq(A, r, rcond=None)[0]
            rk = F["rel_k"]; qs = np.quantile(rk, [0.2, 0.4, 0.6, 0.8])
            b = np.digitize(rk, qs)
            out[f"{arm}|{probe}"] = {
                "matched": float(ok.mean()), "n": int(ok.sum()), "shape": float(r.var()),
                "corr": {k: (float(np.corrcoef(np.nan_to_num(v), r)[0, 1]) if np.std(np.nan_to_num(v)) > 0 else None) for k, v in F.items()},
                "r2_linear": float(1 - np.var(r - A @ co) / r.var()),
                "quintile_mean_by_rel_k": [float(r[b == i].mean()) for i in range(5)],
                "rel_k_quintiles": [float(v) for v in qs]}
    print("__B64__"); print(base64.b64encode(gzip.compress(json.dumps(out).encode())).decode())


if __name__ == "__main__":
    main()
