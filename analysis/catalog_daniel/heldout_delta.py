"""The catalog pretraining (hp58) on the 139 held-out processes of arXiv:2606.23791's split, per event (D74; data from
scripts/job_catalog_heldout_preds.sh via collect_heldout_preds.py -> data/heldout_preds.npz). The parent standardizes
per process, so a held-out process has a prediction in log|M|^2 itself only when it relabels a training process (its
twin's statistics, analysis/catalog_daniel/heldout_twins.json); the 43 others are scored after the best affine map
a + b h only. Delta = |M|^2_pred / |M|^2 - 1 = exp(q - t) - 1 (natural-log target).
  heldout_delta_{a,b}   fraction of events with |Delta| below x, one thin line per process (a: raw, the 96
                        relabels; b: after the affine map, all 139), the median process bold
  heldout_mse_{a,b}     MSE(log|M|^2) per process against its test group, as in arXiv:2606.23791's held-out figure
                        (a: interpolation groups, b: extrapolation groups), raw where it exists and after the map
    python analysis/catalog_daniel/heldout_delta.py
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps  # noqa: E402

X = np.logspace(-6, 1, 141)


def cdf(d):
    d = np.sort(np.abs(d[np.isfinite(d)]))
    return np.searchsorted(d, X, side="right") / len(d)


def main():
    z = np.load(os.path.join(HERE, "data", "heldout_preds.npz"))
    probes = sorted({k.split("__")[0] for k in z.files})
    twins = json.load(open(os.path.join(HERE, "heldout_twins.json")))
    meta = {"dc_" + p["folder"]: p for p in json.load(open(os.path.join(HERE, "classes_default.json")))["processes"]}
    groups = {sp: sorted({meta[p]["test_group"] for p in probes if meta[p]["split"] == sp})
              for sp in ("interpolation", "extrapolation")}

    figs = ps.panels(2)
    for (fig, ax), kind in zip(figs, ("q", "q_aff")):
        for rel, c in ((True, ps.C.blue), (False, ps.C.vermillion)):
            P = [p for p in probes if (p in twins) == rel and f"{p}__{kind}" in z.files]
            if not P:
                continue
            C = np.array([cdf(np.exp(z[f"{p}__{kind}"].astype(float) - z[f"{p}__t"].astype(float)) - 1) for p in P])
            for row in C:
                ax.plot(X, row, "-", color=c, alpha=0.15, lw=0.6)    # weight encodes "one process" vs "median"
            ax.plot(X, np.median(C, 0), "-", color=c, label=f"{'relabel' if rel else 'no relabel'} ({len(P)}), median")
        ax.set_xscale("log"); ax.set_ylim(0, 1)
        ax.set_xlabel(r"$|\Delta|$"); ax.set_ylabel(r"fraction of events below")
        ps.process_label(ax, "raw" if kind == "q" else "after affine map", loc="lower right")
    ax = figs[1][1]
    ax.plot([], [], "-", color="grey", alpha=0.3, lw=0.6, label="one process")
    ps.legend(ax, "upper left")
    ps.save_panels(figs, os.path.join(HERE, "figs", "heldout_delta"))

    figs = ps.panels(2)
    rng = np.random.default_rng(0)
    for (fig, ax), sp in zip(figs, ("interpolation", "extrapolation")):
        for gi, g in enumerate(groups[sp], 1):
            P = [p for p in probes if meta[p]["split"] == sp and meta[p]["test_group"] == g]
            for p in P:
                t = z[f"{p}__t"].astype(float)
                x = gi + rng.uniform(-0.25, 0.25)
                ax.plot([x], [np.mean((z[f"{p}__q_aff"] - t) ** 2)], "o", color=ps.C.blue, mfc="none")
                if f"{p}__q" in z.files:
                    ax.plot([x], [np.mean((z[f"{p}__q"] - t) ** 2)], "o", color=ps.C.vermillion)
        ax.set_yscale("log")
        ax.set_xticks(range(1, len(groups[sp]) + 1))
        ax.set_xlabel(f"{sp} group"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ax = figs[0][1]
    ax.plot([], [], "o", color=ps.C.vermillion, label="raw (relabels)")
    ax.plot([], [], "o", color=ps.C.blue, mfc="none", label="after affine map")
    ps.legend(ax, "upper left")
    ps.save_panels(figs, os.path.join(HERE, "figs", "heldout_mse"))
    for sp in groups:
        print(sp, "groups:", "; ".join(f"{i} {g}" for i, g in enumerate(groups[sp], 1)))
    for kind in ("q", "q_aff"):
        P = [p for p in probes if f"{p}__{kind}" in z.files]
        m = {p: float(np.mean((z[f"{p}__{kind}"] - z[f"{p}__t"]) ** 2)) for p in P}
        md = {p: float(np.median(np.abs(np.exp(z[f"{p}__{kind}"].astype(float) - z[f"{p}__t"].astype(float)) - 1)))
              for p in P}
        for rel in (True, False):
            Q = [p for p in P if (p in twins) == rel]
            if Q:
                print(f"{kind} {'relabel' if rel else 'no relabel'} ({len(Q)}): MSE median {np.median([m[p] for p in Q]):.3g}"
                      f" (10-90% {np.quantile([m[p] for p in Q], .1):.3g} to {np.quantile([m[p] for p in Q], .9):.3g}),"
                      f" median |Delta| per process median {np.median([md[p] for p in Q]):.3g}")
        for p in sorted(P, key=lambda p: -m[p])[:6]:
            print(f"   worst {kind}: {p} MSE {m[p]:.3g} median|Delta| {md[p]:.3g} ({meta[p]['test_group']})")


if __name__ == "__main__":
    main()
