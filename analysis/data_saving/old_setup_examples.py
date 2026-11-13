"""Data saving in the old (transfer-ladder) setup, four examples for the draft (D433): from scratch against fine-tuned,
MSE of log|M|^2 against the fine-tune events D, on the ladder's 8k-step grid (the draft's ladder figures' cells).
Per panel: scratch and the fine-tune from the pretraining with every structure (rung 9 plus every star arm's
processes, "finale" in analysis/transfer), with the target precision 1e-4, or 1e-2 where an arm does not reach 1e-4
(Joaquin 2026-11-13); the saving D_scratch/D_fine-tune is read where each curve first crosses it (log-log interpolation). Cells are read exactly as
analysis/transfer/finale_synth.py reads them (cell(): the best of a search's first 5 trials, a fixed run at the
reference point, or the two chosen points of the W-pole arm); scratch is the 8-trial search (cells.scratch).
    python analysis/data_saving/old_setup_examples.py   -> figs/old_setup_examples; per-cell ratios on stdout
"""
import io, contextlib, os, sys
os.environ["TP_USE_32K"] = "0"      # the 8k grid, as the draft's ladder figures
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "analysis", "transfer"))
sys.path.insert(0, ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    from finale_synth import cell, ARMLAB  # noqa: E402
    from rung_grid import LAB  # noqa: E402
    from cells import scratch  # noqa: E402
import plot_style as ps  # noqa: E402

import numpy as np  # noqa: E402

PROBES = ["uubar_Zggg", "ee_ddbarg", "udbar_enu", "ee_bb_nlo"]
K = range(2, 9)
D = [10 ** (k / 2) for k in K]


def cross(xs, ys, level):
    """D where a curve first reaches `level`, interpolated linearly in log D, log L between the bracketing grid
    points; None if it never does."""
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 > level >= y1:
            f = (np.log(y0) - np.log(level)) / (np.log(y0) - np.log(y1))
            return float(np.exp(np.log(x0) + f * (np.log(x1) - np.log(x0))))
    return None


fig, axs = ps.figure(ncols=2, nrows=2)
ALL = []
for ax, p in zip(axs.flat, PROBES):
    sc = [scratch(p, k, steered=False)[0] for k in K]
    ft = [cell("tp3_finfte", p, k) for k in K]
    curves = {}
    for vals, kw in ((sc, dict(color="k", label="scratch")), (ft, dict(color=ps.C.blue, label="Fine-tune"))):
        pts = [(x, y) for x, y in zip(D, vals) if y]
        curves[kw["label"]] = pts
        ax.plot([q[0] for q in pts], [q[1] for q in pts], "o-", **kw)
    # the line: 1e-4 where both curves reach it, else 1e-2 (Joaquin, 2026-11-13)
    level = 1e-4 if all(min(y for _, y in c) <= 1e-4 for c in curves.values()) else 1e-2
    ax.axhline(level, color="0.5", ls="--", label="target precision")
    ds, df = (cross([q[0] for q in curves[k]], [q[1] for q in curves[k]], level) for k in ("scratch", "Fine-tune"))
    g = [a / b for a, b in zip(sc, ft) if a and b]
    ALL += g
    print(f"{p}: level {level:g}  D_scratch {ds:.0f}  D_ft {df:.0f}  saving D_scratch/D_ft {ds / df:.1f}  "
          f"median L_scratch/L_ft over D {np.median(g):.1f} (range {min(g):.1f}-{max(g):.1f}, {len(g)} cells)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(ax, {**LAB, **ARMLAB}[p].replace(" (1-loop)", " one loop"))
print(f"median L_scratch/L_ft over all {len(ALL)} cells {np.median(ALL):.1f}")
ps.legend(axs.flat[0], "lower left")
for ax in axs.flat:
    ps.make_room(ax)
ps.save(fig, os.path.join(ROOT, "analysis", "data_saving", "figs", "old_setup_examples"))
