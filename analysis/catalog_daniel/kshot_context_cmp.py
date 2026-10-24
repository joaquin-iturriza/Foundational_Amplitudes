"""D105, what the context vector buys: the K-shot fit of c_p (tools/kshot_context.py on DlessInOrdQc, run
FA-ccin2p3-dc3b51b7) against a per-process offset on the same model and the same draws. Per held-out process the saved
npz (data/kshot_context/<set>/<probe>.npz: the c = 0 prediction q0, the truth t, per K the 5 drawn event indices and the
MSE of log|M|^2 on the other events after the fit and after the plain K-event offset) gives, per draw:
  context fit     c fitted on the K events, everything else frozen (the job's msefit)
  plain offset    the mean error of the c = 0 prediction on the K events, subtracted (the job's mseoff)
  shrunk offset   kshot_level.py's estimator on these draws: the prior on the offset is the mean and spread of the other
                  processes' offsets in the same set (leave-one-process-out), combined with the K events by precision
                  (the process's error variance as the noise); K = 0 is the prior alone
and the floor, the variance of the c = 0 error (what any constant per process reaches). Mean over the draws, median over
the processes of a set (the 43 new, the 2->4 probes with a gluon, with a photon); also the share of processes where the
context fit beats the shrunk offset.
    python3 analysis/catalog_daniel/kshot_context_cmp.py   -> data/kshot_context_cmp.json, figs/kshot_context_{a,b,c}
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
KS = [1, 4, 16]
SETS = {"43 new": ("new43", lambda p: True), "2to4 +g": ("2to4", lambda p: p.endswith("_g")),
        "2to4 +a": ("2to4", lambda p: p.endswith("_a"))}


def load(sub, keep):
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, "data", "kshot_context", sub, "*.npz"))):
        p = os.path.basename(f)[:-4]
        if keep(p):
            out[p] = dict(np.load(f))
    return out


def shrunk(d, others, pick):
    mu0, tau2, s2 = others.mean(), others.var() + 1e-9, d.var()
    K = len(pick)
    w = (K / s2) / (K / s2 + 1 / tau2) if K else 0.0
    est = w * (d[pick].mean() if K else 0.0) + (1 - w) * mu0
    rest = np.setdiff1d(np.arange(len(d)), pick)
    return float(np.mean((d[rest] - est) ** 2))


def main():
    res = {}
    for key, (sub, keep) in SETS.items():
        z = load(sub, keep)
        offs = {p: float(np.mean(v["q0"].astype(float) - v["t"].astype(float))) for p, v in z.items()}
        rows = {"zero": [], "floor": [], "shrunk0": []}
        rows.update({f"{m}{K}": [] for m in ("fit", "off", "shr") for K in KS})
        wins = {K: [] for K in KS}
        for p, v in z.items():
            d = v["q0"].astype(float) - v["t"].astype(float)
            oth = np.array([x for q, x in offs.items() if q != p])
            rows["zero"].append(float(np.mean(d ** 2))); rows["floor"].append(float(d.var()))
            rows["shrunk0"].append(shrunk(d, oth, np.array([], dtype=int)))
            for K in KS:
                s = float(np.mean([shrunk(d, oth, pk) for pk in v[f"pick_k{K}"]]))
                f = float(np.mean(v[f"msefit_k{K}"]))
                rows[f"fit{K}"].append(f); rows[f"off{K}"].append(float(np.mean(v[f"mseoff_k{K}"])))
                rows[f"shr{K}"].append(s); wins[K].append(f < s)
        r = {"n": len(z), **{k: float(np.median(x)) for k, x in rows.items()},
             **{f"fit_beats_shrunk{K}": float(np.mean(wins[K])) for K in KS},
             **{f"ratio_fit_shr{K}": [float(x) for x in np.quantile(np.array(rows[f"fit{K}"]) /
                                                                   np.array(rows[f"shr{K}"]), [0.1, 0.5, 0.9])]
                for K in KS}}
        res[key] = r
        print(key, json.dumps(r))
    res["_K"] = KS
    json.dump(res, open(os.path.join(HERE, "data", "kshot_context_cmp.json"), "w"), indent=1)
    plot(res)


def plot(res):
    import plot_style as ps
    figs = ps.panels(3)
    labels = {"43 new": "43 new processes", "2to4 +g": r"$2\to4$, $+g$", "2to4 +a": r"$2\to4$, $+\gamma$"}
    for (fig, ax), key in zip(figs, SETS):
        r = res[key]
        x = [0] + KS
        ax.plot(x, [r["zero"]] + [r[f"off{K}"] for K in KS], marker="o", color=ps.C.blue, label="plain offset")
        ax.plot(x, [r["shrunk0"]] + [r[f"shr{K}"] for K in KS], marker="s", color=ps.C.green, label="shrunk offset")
        ax.plot(x, [r["zero"]] + [r[f"fit{K}"] for K in KS], marker="^", color=ps.C.vermillion,
                label=r"context $c_p$ fit")
        ax.plot([0, KS[-1]], [r["floor"]] * 2, color=ps.C.grey, ls="--", label="floor (error variance)")
        ax.set_xscale("symlog", linthresh=1); ax.set_yscale("log")
        ax.set_xticks([0, 1, 4, 16]); ax.set_xticklabels(["0", "1", "4", "16"])
        ax.set_xlabel(r"labelled events $K$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, labels[key])
    ps.legend(figs[0][1], "lower left")
    for _, ax in figs:
        ps.make_room(ax)
    ps.save_panels(figs, os.path.join(HERE, "figs", "kshot_context"))


if __name__ == "__main__":
    main()
