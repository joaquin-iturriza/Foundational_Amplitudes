"""Transfer study, the finale and the synthetic-amplitude pretraining as learning curves: per probe, the loss (MSE of
log|M|^2 at the best checkpoint) against the fine-tune events D, 8k grid, for scratch, rung 0 (ee->uu), rung 9,
synthetic and finale on the twelve ladder probes; scratch, the star arm and the finale on the six star-arm probes. A
cell is drawn once finished (finale_synth.cell: a search with its 5 trials in, or its fixed run at the reference point).
  figs/finale_synth_scaling_<ee|qcd|arm>_a..f, _legend
    python analysis/transfer/finale_synth_scaling.py
    python analysis/transfer/finale_synth_scaling.py --paper   -> figs/finale_synth_paper_<ee|qcd>: the paper draft's
        version (the user's call, 2026-10-09): scratch, rung 9, synthetic and finale only, scratch drawn as in rung_focus
"""
import io, contextlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, scratch  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    from finale_synth import ARMP, ARMLAB, LAB, P, cell  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

LADDER = [("tp3_uu64fte", "rung 0", ps.C.sky, "o"), ("tp3_r9fte", "rung 9", ps.C.blue, "s"),
          ("tp3_synfte", "synthetic", ps.C.green, "^"), ("tp3_finfte", "finale", ps.C.vermillion, "D")]
FIGS = os.path.join(ROOT, "analysis", "transfer", "figs")
PAPER = "--paper" in sys.argv
BASE = "finale_synth_paper" if PAPER else "finale_synth_scaling"
SCR = dict(color="k", marker="o", label="from scratch") if PAPER else dict(color="k", ls="--", marker="x", label="scratch")


def curve(ax, xs, ys, **kw):
    pts = [(x, y) for x, y in zip(xs, ys) if y]
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], **kw)


def draw(part, probes, fams, labels):
    figs = ps.panels(len(probes))
    for (fig, ax), p in zip(figs, probes):
        K = list(range(2, 9))
        D = [10 ** (k / 2) for k in K]
        curve(ax, D, [scratch(p, k, steered=False)[0] for k in K], **SCR)
        for f, lab, c, m in fams(p):
            curve(ax, D, [cell(f, p, k) for k in K], color=c, marker=m, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, labels[p])
    a0 = max((ax for _, ax in figs), key=lambda ax: len(ax.get_legend_handles_labels()[1]))   # the fullest panel
    ps.legend_strip(a0, os.path.join(FIGS, f"{BASE}_{part}_legend"), ncol=5)
    ps.save_panels(figs, os.path.join(FIGS, f"{BASE}_{part}"))


fams = (lambda p: LADDER[1:]) if PAPER else (lambda p: LADDER)
draw("ee", P[:6], fams, LAB)
draw("qcd", P[6:], fams, LAB)
if not PAPER:
    draw("arm", list(ARMP), lambda p: [(ARMP[p][0], "the probe's star arm", ps.C.orange, "o"),
                                     ("tp3_finfte", "finale", ps.C.vermillion, "D")], ARMLAB)
