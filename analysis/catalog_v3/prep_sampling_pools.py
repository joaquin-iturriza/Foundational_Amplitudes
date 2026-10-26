"""The sampling sensitivity on catalog_v3 (the preprocessing exploration, D83/D114; the port of the item of
par. More data-free normalizations, and a learned level, analysis/catalog_daniel/norm_explore6.py). How much of what a
target and a trained network show depends on how the pool drew its events? Two reweightings of each pool's events:
    flat    flat in log s_hat: 20 bins over the 1 to 99 percentile range of the events' log s_hat (the events beyond it
            in the end bins), every non-empty bin the same weight (the definition behind the notes' 0.30 to 0.39,
            analysis/catalog_daniel/norm_moments.py and norm_explore.py); catalog_v4 draws sqrt(s_hat) log-uniform, so
            this is catalog_v4's measure inside catalog_v3's window
    low     the 20% of events closest to a singular region (lowest Lmin, the least pair invariant over s_hat, the beam
            pair excluded) carry half the weight
Data part, per class on its whole validation pool (10^4 events; positive pools only), per target (raw log|M|^2;
dless + (N - 4) log s_hat; qperp + the emission scales; soft, the same with tau = 0.3; prep_shape.py's four): how far the
level moves under each reweighting in units of the pool's std (shift_<w>), and the std's ratio (sd_<w>).
Network part, per held-out probe and short arm (scripts/job_catalog_v3_short.sh; its saved predictions
$SCRATCH/zz_v3_short_<arm>/<axis>/<probe>.npz, truth t and prediction q of log|M|^2 on the first 2000 validation events):
the zero-shot MSE, level and shape under uniform, flat and low weights on those events (the read shuffles them; each is
matched to its pool event by sorting both on log|M|^2). align = the largest abs(t - logF - log abs(M)^2) over the matched
events off the floor (t is log(|M|^2 F), logF saved beside it); an arm
whose align exceeds ALIGN_TOL (float32 rounding of a log of order 10 is 1e-6) or that saved no prediction reports why
instead of numbers.
Sets: the 200 training classes of prep_shape.py (seed 0) and every class of the held-out axes H1 H2 H2p5 H3 H5 H6 H7c H7E.
    python analysis/catalog_v3/prep_sampling_pools.py [--workers 32]   (a CPU job on CC: scripts/job_prep_sampling_v3.sh)
Output: one SAMPLING line per class in the log, nothing written to disk.
"""
import argparse, glob, json, os, sys
from multiprocessing import Pool

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from physics_scales import minkowski_dots, qperp_logs  # noqa: E402

AXES = ("H1", "H2", "H2p5", "H3", "H5", "H6", "H7c", "H7E")
ARMS = ("baseline", "dimensionless", "dimensionless_shat", "dimensionless_hybrid", "dimensionless_blend025",
        "dimensionless_blend050", "dimensionless_context", "dimensionless_twin")
NBINS = 20
ALIGN_TOL = 1e-4


def pool(name, sq):
    d = os.environ["DATA_DIR"]
    g = sorted(glob.glob(os.path.join(d, f"{name}_{sq[0]}-{sq[1]}GeV_val_*amplitudes.npy"))) if sq else []
    g = g or sorted(glob.glob(os.path.join(d, f"{name}_*_val_*amplitudes.npy")))
    return g[0] if g else None


def w_flat(L, nb=NBINS):
    lo, hi = np.percentile(L, [1, 99])
    b = np.clip(((L - lo) / max(hi - lo, 1e-9) * nb).astype(int), 0, nb - 1)
    cnt = np.bincount(b, minlength=nb)
    w = 1.0 / cnt[b]
    return w / w.sum()


def w_low(Lm):
    low = Lm <= np.quantile(Lm, 0.2)
    return np.where(low, 0.5 / low.sum(), 0.5 / (~low).sum())


def r4(x):
    return float(f"{x:.4g}")


def one(args):
    setname, name, sq = args
    f = pool(name, sq)
    if f is None:
        return setname, name, "no pool"
    x = np.asarray(np.load(f, mmap_mode="r"), dtype=np.float64)
    N = (x.shape[1] - 1) // 5
    P = x[:, : 4 * N].reshape(len(x), N, 4); amp = x[:, -1]
    if not np.all(np.isfinite(amp)):
        return setname, name, "non-finite"
    G = minkowski_dots(P)
    Ls = np.log(np.abs(G[:, 0, 0] + G[:, 1, 1] + 2 * G[:, 0, 1]))
    iu = [(i, j) for i in range(N) for j in range(i + 1, N) if (i, j) != (0, 1)]
    S = np.stack([np.log(np.clip(2 * np.abs(G[:, i, j]), 1e-300, None)) for i, j in iu], 1)
    Lm = S.min(1) - Ls
    signed = bool(np.any(amp <= 0))
    out = {"N": int(N), "n": int(len(x)), "signed": signed, "logshat_range": r4(np.ptp(Ls))}
    W = {"uni": np.full(len(x), 1 / len(x)), "flat": w_flat(Ls), "low": w_low(Lm)}
    if not signed:
        y0 = np.log(amp)
        tg = {"raw": y0, "dless": y0 + (N - 4) * Ls, "qperp": y0 + qperp_logs(P),
              "soft": y0 + qperp_logs(P, soft=0.3)}
        for k, y in tg.items():
            mu, sd = y.mean(), y.std()
            d = {}
            for wn in ("flat", "low"):
                w = W[wn]; m = np.sum(w * y)
                d[f"shift_{wn}"] = r4(abs(m - mu) / sd)
                d[f"sd_{wn}"] = r4(np.sqrt(np.sum(w * (y - m) ** 2)) / sd)
            out[k] = d
    if setname != "train":
        la = np.log(np.clip(np.abs(amp), 1e-300, None))
        net = {}
        for arm in ARMS:
            g = os.path.join(os.environ.get("SCRATCH", "/tmp"), f"zz_v3_short_{arm}", setname, name + ".npz")
            if not os.path.exists(g):
                continue
            z = np.load(g)
            if "q" not in z.files:
                continue
            t, q = z["t"].astype(np.float64), z["q"].astype(np.float64)
            n = len(t)
            if len(q) != n:
                net[arm] = {"why": "no prediction (q empty)"}
                continue
            tl = t - z["logF"].astype(np.float64) if len(z["logF"]) == n else t
            # the read loads the pool's first n events and shuffles them: event k of the saved arrays is pool event
            # perm[k], found by sorting both on log|M|^2 (ties only among events at the floor)
            perm = np.empty(n, int); perm[np.argsort(tl, kind="stable")] = np.argsort(la[:n], kind="stable")
            ok = amp[perm] != 0
            if "at_floor" in z.files and len(z["at_floor"]) == n:
                ok &= ~z["at_floor"]
            al = float(np.max(np.abs(tl[ok] - la[perm][ok]))) if ok.any() else np.inf
            if not al <= ALIGN_TOL:
                net[arm] = {"why": "not aligned", "align": r4(al) if np.isfinite(al) else None}
                continue
            e = q - t
            r = {"align": r4(al)}
            ww = {"uni": np.full(n, 1 / n), "flat": w_flat(Ls[perm]), "low": w_low(Lm[perm])}
            for wn, w in ww.items():
                mse = float(np.sum(w * e ** 2)); lev = float(np.sum(w * e)) ** 2
                r[wn] = [r4(mse), r4(lev), r4(mse - lev)]
            net[arm] = r
        out["net"] = net
    return setname, name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--n-train-classes", type=int, default=200)
    a = ap.parse_args()
    tr = [p for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", "catalog_v3_train.yaml")))["processes"]
          if p["name"].startswith("v3_") and "_nlo" not in p["name"] and "_loop" not in p["name"]]
    pick = np.random.default_rng(0).permutation(len(tr))[: a.n_train_classes]
    jobs = [("train", tr[i]["name"], tr[i].get("sqrts")) for i in sorted(pick)]
    for ax in AXES:
        for p in yaml.safe_load(open(os.path.join(ROOT, "recipes", f"catalog_v3_heldout_{ax}.yaml")))["processes"]:
            jobs.append((ax, p["name"], p.get("sqrts")))
    skipped = []
    with Pool(a.workers) as pl:
        for s, n, r in pl.imap_unordered(one, jobs, chunksize=1):
            if isinstance(r, str):
                skipped.append((s, n, r))
                print("SAMPLING_SKIP", json.dumps({"set": s, "name": n, "why": r}), flush=True)
                continue
            print("SAMPLING", json.dumps({"set": s, "name": n, **r}, separators=(",", ":")), flush=True)
    print("SAMPLING_DONE", len(jobs), "classes,", len(skipped), "skipped", flush=True)


if __name__ == "__main__":
    main()
