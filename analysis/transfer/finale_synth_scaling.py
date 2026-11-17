"""Transfer study, the finale and the synthetic-amplitude pretraining as learning curves: per probe, the loss (MSE of
log|M|^2 at the best checkpoint) against the fine-tune events D, 8k grid, for scratch, rung 0 (ee->uu), rung 9,
synthetic and finale on the twelve ladder probes; scratch, the star arm and the finale on the six star-arm probes. A
cell is drawn once finished (finale_synth.cell: a search with its 5 trials in, or its fixed run at the reference point).
  figs/finale_synth_scaling_<ee|qcd|arm> (one file each)
    python analysis/transfer/finale_synth_scaling.py
    python analysis/transfer/finale_synth_scaling.py --paper   -> figs/finale_synth_paper_<ee|qcd>: the paper draft's
        version: scratch, the rung below the probe's structure and the first rung with it (rung_grid.FOCUS), synthetic,
        and the finale named by what it is, "every structure" (D434: before/after kept, no forced rung 9; was rung 9,
        synthetic and finale, the user's call 2026-10-09)
"""
import io, contextlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, scratch  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    from finale_synth import ARMP, ARMLAB, LAB, P, cell  # noqa: E402
    from rung_grid import FOCUS, fam  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

LADDER = [("tp3_uu64fte", "rung 0", ps.C.sky, "o"), ("tp3_r9fte", "rung 9", ps.C.blue, "s"),
          ("tp3_synfte", "synthetic", ps.C.green, "^"), ("tp3_finfte", "finale", ps.C.vermillion, "D")]
FIGS = os.path.join(ROOT, "analysis", "transfer", "figs")
PAPER = "--paper" in sys.argv
BASE = "finale_synth_paper" if PAPER else "finale_synth_scaling"
SCR = dict(color="k", marker="o", label="from scratch", zorder=10) if PAPER else dict(color="k", ls="--", marker="x", label="scratch")


def curve(ax, xs, ys, **kw):
    pts = [(x, y) for x, y in zip(xs, ys) if y]
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], **kw)


def draw(part, probes, fams, labels):
    """One file per figure (Joaquin 2026-11-17), six panels in three rows of two, each with its own legend where its
    curves differ from the others' (the paper's steps), else one legend on the first panel."""
    fig, axs = ps.figure(ncols=2, nrows=3)
    for i, (ax, p) in enumerate(zip(axs.flat, probes)):
        K = list(range(2, 9))
        D = [10 ** (k / 2) for k in K]
        curve(ax, D, [scratch(p, k, steered=False)[0] for k in K], **SCR)
        for f, lab, c, m in fams(p):
            curve(ax, D, [cell(f, p, k) for k in K], color=c, marker=m, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"fine-tuning events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, labels[p])
        if PAPER or i == 0:
            ps.legend(ax, "lower left", fontsize=ps.LEGEND_MIN_PT)
    ps.save(fig, os.path.join(FIGS, f"{BASE}_{part}"))


if PAPER:
    def upto(r):    # as rung_focus.upto (importing rung_focus draws its figures)
        return r"$e^+e^-\to u\bar u$ only" if r == 0 else ("ladder step 1" if r == 1 else f"ladder steps 1-{r}")

    def fams(p):
        r = FOCUS[p]
        return [(fam(r - 1), upto(r - 1), ps.C.blue, "o"), (fam(r), upto(r), ps.C.vermillion, "s"),
                ("tp3_synfte", "synthetic amplitudes", ps.C.green, "^"),
                ("tp3_finfte", "ladder steps 1-9 + every star arm", ps.C.orange, "D")]
else:
    fams = lambda p: LADDER  # noqa: E731
draw("ee", P[:6], fams, LAB)
draw("qcd", P[6:], fams, LAB)
if not PAPER:
    draw("arm", list(ARMP), lambda p: [(ARMP[p][0], "the probe's star arm", ps.C.orange, "o"),
                                     ("tp3_finfte", "finale", ps.C.vermillion, "D")], ARMLAB)
