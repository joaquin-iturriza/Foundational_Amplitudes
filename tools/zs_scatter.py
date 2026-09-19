#!/usr/bin/env python3
"""
zs_scatter.py -- Predicted-vs-true log-amplitude density (hexbin) for the
zero-shot held-out processes, one figure per process and one panel per model.
Horizontal banding means the model predicts about the same value whatever the
true amplitude (under-dispersion). Prints the per-cell correlation and the
ratio of predicted to true spread for the caption.

Reads analysis/zero_shot/<label>__<dataset>_arrays.npz (tools/zero_shot_eval.py).
Usage: python tools/zs_scatter.py
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402
from zs_hist import cells, PROCS  # noqa: E402

ZS = os.path.join(ROOT, "analysis", "zero_shot")


def draw(ax, mlabel, t, p, proc):
    lo = min(t.min(), p.min()); hi = max(t.max(), p.max())
    pad = 0.05 * (hi - lo); lo -= pad; hi += pad
    hb = ax.hexbin(t, p, gridsize=45, bins="log", cmap="viridis",
                   extent=(lo, hi, lo, hi), mincnt=1)
    ax.plot([lo, hi], [lo, hi], "--", color=ps.C.vermillion, label=r"$y=x$")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"true $\log|\mathcal{M}|^2$")
    ax.set_ylabel(r"predicted $\log|\mathcal{M}|^2$")
    ps.colorbar(ax, hb, "events")
    ps.legend(ax, "upper left")
    ps.process_label(ax, proc, loc="lower right")
    rho = np.corrcoef(t, p)[0, 1]
    print(f"  {proc} {mlabel}: rho={rho:.2f} sigma_pred/sigma_true={p.std()/t.std():.2f}")


def main():
    for ds, proc, tag in PROCS:
        cs = cells(ds)
        base = os.path.join(ZS, f"zero_shot_scatter_{tag}")
        if len(cs) == 2:
            fig, axes = ps.figure(ncols=2)
            for ax, (m, t, p) in zip(axes, cs):
                draw(ax, m, t, p, proc)
            ps.save(fig, base)
        else:
            figs = ps.panels(len(cs))
            for (fig, ax), (m, t, p) in zip(figs, cs):
                draw(ax, m, t, p, proc)
            ps.save_panels(figs, base)


if __name__ == "__main__":
    main()
