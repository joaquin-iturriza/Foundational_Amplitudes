"""The context vector's K-shot fits on catalog_v3 (scripts/job_kshot_context_v3.sh on the short run with the context
vector; its npz copied home into data/kshot_v3/<axis>/<probe>.npz), read as on Daniel's catalog
(analysis/catalog_daniel/kshot_context_cmp.py): per held-out process and K = 1, 4, 16 (5 draws each), the MSE of
log|M|^2 on the other validation events
  context fit     c fitted on the K events, everything else frozen
  plain offset    the mean error of the c = 0 prediction on the K events, subtracted
  shrunk offset   that offset pulled toward the other processes' offsets on the same axis (leave-one-process-out), by
                  precision; K = 0 is the prior alone
and the floor, the variance of the c = 0 error (what any constant per process reaches). Mean over the draws, median
over an axis's processes, the share of processes where the fit beats the shrunk offset, and every expected process
without a result listed (recipes/catalog_v3_heldout_<axis>.yaml). Figure: the same curves pooled over the axes inside
the catalog's range and over the two beyond it (2->5, 1-10 TeV).
    python3 analysis/catalog_v3/kshot_context.py   -> data/kshot_context.json, figs/kshot_context
"""
import glob, json, os, sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis", "catalog_daniel"))
from kshot_context_cmp import KS, shrunk  # noqa: E402

AXES = ("H1", "H2", "H3", "H5", "H6", "H7c", "H2p5", "H7E")
BEYOND = ("H2p5", "H7E")


def per_process(axis):
    z = {}
    for f in sorted(glob.glob(os.path.join(HERE, "data", "kshot_v3", axis, "*.npz"))):
        z[os.path.basename(f)[:-4]] = dict(np.load(f))
    offs = {p: float(np.mean(v["q0"].astype(float) - v["t"].astype(float))) for p, v in z.items()}
    out = {}
    for p, v in z.items():
        d = v["q0"].astype(float) - v["t"].astype(float)
        oth = np.array([x for q, x in offs.items() if q != p])
        r = {"zero": float(np.mean(d ** 2)), "floor": float(d.var()), "shrunk0": shrunk(d, oth, np.array([], dtype=int))}
        for K in KS:
            r[f"fit{K}"] = float(np.mean(v[f"msefit_k{K}"]))
            r[f"off{K}"] = float(np.mean(v[f"mseoff_k{K}"]))
            r[f"shr{K}"] = float(np.mean([shrunk(d, oth, pk) for pk in v[f"pick_k{K}"]]))
        out[p] = r
    return out


def summary(rows):
    keys = list(next(iter(rows.values())))
    s = {k: float(np.median([r[k] for r in rows.values()])) for k in keys}
    for K in KS:
        s[f"fit_beats_shrunk{K}"] = float(np.mean([r[f"fit{K}"] < r[f"shr{K}"] for r in rows.values()]))
    s["n"] = len(rows)
    return s


def main():
    res = {}
    pooled = {"inside": {}, "beyond": {}}
    for ax in AXES:
        exp = [e["name"] for e in yaml.safe_load(open(os.path.join(ROOT, "recipes", f"catalog_v3_heldout_{ax}.yaml")))["processes"]]
        rows = per_process(ax)
        missing = sorted(set(exp) - set(rows))
        if not rows:
            res[ax] = {"n": 0, "n_expected": len(exp), "missing": missing}
            print(f"{ax}: no results copied home")
            continue
        res[ax] = {**summary(rows), "n_expected": len(exp), "missing": missing, "processes": rows}
        pooled["beyond" if ax in BEYOND else "inside"].update({f"{ax}/{p}": r for p, r in rows.items()})
        s = res[ax]
        print(f"{ax:5s} {s['n']:3d}/{len(exp):3d} zero {s['zero']:.3g} floor {s['floor']:.3g} | K=16 off {s['off16']:.3g} "
              f"shr {s['shr16']:.3g} fit {s['fit16']:.3g} fit<shr {s['fit_beats_shrunk16']:.2f}")
    res["pooled"] = {k: summary(v) for k, v in pooled.items() if v}
    json.dump(res, open(os.path.join(HERE, "data", "kshot_context.json"), "w"), indent=1)
    plot(res["pooled"])


def plot(pooled):
    import plot_style as ps
    fig, axs = ps.figure(ncols=2)
    labels = {"inside": "within the catalog's range", "beyond": r"$2\to5$ and $1$-$10$ TeV"}
    for ax, key in zip(axs, ("inside", "beyond")):
        if key not in pooled:
            continue
        r = pooled[key]
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
    ps.legend(axs[0], "lower left")
    for ax in axs:
        ps.make_room(ax)
    ps.save(fig, os.path.join(HERE, "figs", "kshot_context"))


if __name__ == "__main__":
    main()
