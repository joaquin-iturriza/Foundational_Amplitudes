"""How much do K labelled events of a process fix a zero-shot prediction? (2026-10-23; the user: "just having one
datapoint from the dataset already helps a ton".) For every held-out set of the dimensionless round (D93), per process,
K events are drawn at random from its scored validation events, the offset is estimated as the mean error on them, and
the MSE of log|M|^2 is measured on the other events after subtracting it; 200 draws per (process, K), the median over
processes reported. K = 0 is the zero-shot read; K -> infinity tends to the shape error (the error's variance), the floor
any per-process constant can reach. Also a shrinkage estimate: the prior on a process's offset is the mean and spread
of the other processes' offsets in the same set (the axis' systematic bias, leave-one-process-out), combined with the K
events by precision (the error's variance on the process as the noise); at K = 0 it is the axis bias alone. Also the
K-event affine fit (offset and scale, K >= 3), which tells whether the
spread is off as well as the level.
Inputs: the npz of the held-out reads in data/ (collect_heldout_preds.py). Output: data/kshot_level.json, the table
printed, figs/kshot_level.{png,pdf}.
    .venv-driver/bin/python analysis/catalog_daniel/kshot_level.py
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
KS = [0, 1, 2, 4, 8, 16, 32, 64]
LOOP = ("_nlo_", "_loop_")


def sets():
    twins = set(json.load(open(os.path.join(HERE, "heldout_twins.json"))))
    out = {}
    for arm in ("DlessIn", "DlessInOrd"):
        z = np.load(os.path.join(HERE, "data", f"catalog_short_{arm}_sigstage.npz"))
        ps = sorted({k.split("__")[0] for k in z.files if k.endswith("__q")})
        out[f"43 new|{arm}"] = (z, [p for p in ps if p not in twins])
        z4 = np.load(os.path.join(HERE, "data", f"catalog_2to4_short_{arm}.npz"))
        ps4 = sorted({k.split("__")[0] for k in z4.files if k.endswith("__q")})
        out[f"2to4 +g|{arm}"] = (z4, [p for p in ps4 if p.endswith("_g")])
        out[f"2to4 +a|{arm}"] = (z4, [p for p in ps4 if p.endswith("_a")])
        zl = np.load(os.path.join(HERE, "data", f"catalog_loop_short_{arm}.npz"))
        psl = sorted({k.split("__")[0] for k in zl.files if k.endswith("__q")})
        out[f"loop trees|{arm}"] = (zl, [p for p in psl if not any(s in p for s in LOOP)])
        out[f"loop one-loop|{arm}"] = (zl, [p for p in psl if any(s in p for s in LOOP)])
    zt = np.load(os.path.join(HERE, "data", "catalog_top_short_DlessInOrd_notop.npz"))
    out["top, never seen|DlessInOrd"] = (zt, sorted({k.split("__")[0] for k in zt.files if k.endswith("__q")}))
    return out


def kshot(d, K, rng, n=200):
    """the mean MSE over draws: the expected error after a K-event offset correction"""
    if K == 0:
        return float(np.mean(d ** 2))
    m = []
    for _ in range(n):
        i = rng.permutation(len(d)); c, r = i[:K], i[K:]
        m.append(np.mean((d[r] - d[c].mean()) ** 2))
    return float(np.mean(m))


def kshot_shrink(d, others, K, rng, n=100):
    mu0, tau2, s2 = others.mean(), others.var() + 1e-9, d.var()
    w = (K / s2) / (K / s2 + 1 / tau2) if K else 0.0
    m = []
    for _ in range(n):
        i = rng.permutation(len(d)); c, r = i[:K], i[K:]
        est = w * (d[c].mean() if K else 0.0) + (1 - w) * mu0
        m.append(np.mean((d[r] - est) ** 2))
    return float(np.mean(m))


def kshot_affine(q, t, K, rng, n=200):
    if K < 3:
        return np.nan
    m = []
    for _ in range(n):
        i = rng.permutation(len(t)); c, r = i[:K], i[K:]
        A = np.stack([np.ones(K), q[c]], 1)
        co = np.linalg.lstsq(A, t[c], rcond=None)[0]
        m.append(np.mean((co[0] + co[1] * q[r] - t[r]) ** 2))
    return float(np.mean(m))


def main():
    rng = np.random.default_rng(0)
    res = {}
    for key, (z, ps) in sets().items():
        rows = {K: [] for K in KS}; aff = {K: [] for K in KS}; shr = {K: [] for K in KS}; shape = []
        offs = {p: float(np.mean(z[f"{p}__q"].astype(float) - z[f"{p}__t"].astype(float))) for p in ps}
        for p in ps:
            q, t = z[f"{p}__q"].astype(float), z[f"{p}__t"].astype(float)
            d = q - t
            shape.append(d.var())
            for K in KS:
                rows[K].append(kshot(d, K, rng)); aff[K].append(kshot_affine(q, t, K, rng))
                shr[K].append(kshot_shrink(d, np.array([v for x, v in offs.items() if x != p]), K, rng))
        r = dict(n=len(ps), shape=float(np.median(shape)),
                 offset=[float(np.median(rows[K])) for K in KS], shrink=[float(np.median(shr[K])) for K in KS],
                 affine=[float(np.nanmedian(aff[K])) if K >= 3 else None for K in KS])
        res[key] = r
        print(f"{key:30s} n={r['n']:3d} shape {r['shape']:.3g} | " +
              " ".join(f"K{K}:{v:.3g}" for K, v in zip(KS, r["offset"])) + " | affine " +
              " ".join(f"K{K}:{v:.3g}" for K, v in zip(KS, r["affine"]) if v is not None) + " | axis prior " +
              " ".join(f"K{K}:{v:.3g}" for K, v in zip(KS, r["shrink"])))
    res["_K"] = KS
    json.dump(res, open(os.path.join(HERE, "data", "kshot_level.json"), "w"), indent=1)
    plot(res)


LABELS = {"43 new": "43 new processes", "2to4 +g": r"$2\to4$, $+g$", "2to4 +a": r"$2\to4$, $+\gamma$",
          "loop trees": "trees of the loop probes", "loop one-loop": "one-loop pools", "top, never seen": "top, never seen"}
COLS = {"43 new": "blue", "2to4 +g": "vermillion", "2to4 +a": "orange", "loop trees": "green", "loop one-loop": "purple",
        "top, never seen": "sky"}


def plot(res):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    for ax, arm in zip(axs, ("DlessIn", "DlessInOrd")):
        for s, lab in LABELS.items():
            r = res.get(f"{s}|{arm}")
            if not r:
                continue
            ax.plot(KS, r["offset"], marker="o", color=getattr(ps.C, COLS[s]), label=lab)
        ax.set_xscale("symlog", linthresh=1); ax.set_yscale("log")
        ax.set_xlabel(r"labelled events $K$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ax.set_xticks([0, 1, 4, 16, 64]); ax.set_xticklabels(["0", "1", "4", "16", "64"])
    ps.shared_legend(fig, axs[1], ncol=3)
    ps.save(fig, os.path.join(HERE, "figs", "kshot_level"))


if __name__ == "__main__":
    main()
