"""Transfer study, the fine-tune grid: each probe's loss against D, from scratch and fine-tuned from each pretraining
(ee->uu, 64k steps, as rung 0, and ladder rungs 1-9; tp3_<parent>fte: training.lr searched over [1e-3, 1e-2],
lr_scale = layer_decay = 1; docs/results.tex sec:ladder), the study's final setup only. A cell's value is its search's
best (MSE of log|M|^2 at the best checkpoint). With cells.USE_32K off (now) every cell is the 8k grid, every
pretraining at equal compute; on, the 32k cells at D = 10^3.5, 10^4 replace them (cells.final), a cell still at 8k drawn open. ee->WW on the mixture pool.
  <base>_ee_a..f, <base>_qcd_a..f
    python analysis/transfer/rung_grid.py      -> figs/rung_grid
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, ARM, USE_32K, best, scratch, final  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

LAB = {"ee_ddbar": r"$e^+e^-\to d\bar d$", "ee_nnbar": r"$e^+e^-\to\nu_e\bar\nu_e$", "ee_ttbar": r"$e^+e^-\to t\bar t$",
       "ee_WW": r"$e^+e^-\to W^+W^-$", "ee_dd_nlo": r"$e^+e^-\to d\bar d$ (1-loop)",
       "ee_bb_nlo": r"$e^+e^-\to b\bar b$ (1-loop)", "ee_Za": r"$e^+e^-\to Z\gamma$", "ud_ud": r"$ud\to ud$",
       "uubar_gg": r"$u\bar u\to gg$", "uubar_Zg": r"$u\bar u\to Zg$", "uubar_Zgg": r"$u\bar u\to Zgg$",
       "uubar_Zggg": r"$u\bar u\to Zggg$",
       "ee_ddbarg": r"$e^+e^-\to d\bar dg$", "ee_ttbarg": r"$e^+e^-\to t\bar tg$",
       "ee_ttbar_nlo_thr": r"$e^+e^-\to t\bar t$ (1-loop, thr.)", "ee_dd_nlo_hi": r"$e^+e^-\to d\bar d$ (1-loop, high)",
       "udbar_enu": r"$u\bar d\to e^+\nu_e$", "ee_dd_ew_nlo": r"$e^+e^-\to d\bar d$ (EW 1-loop)"}
P = ["ee_ddbar", "ee_nnbar", "ee_ttbar", "ee_WW", "ee_dd_nlo", "ee_bb_nlo", "ee_Za", "ud_ud", "uubar_gg",
     "uubar_Zg", "uubar_Zgg", "uubar_Zggg"]
# what each ladder rung adds (recipes/transfer_ladder_r*.yaml headers); the rungs are cumulative
RUNG_ADDS = {1: r"massless $s$-channel ($\gamma/Z$)", 2: r"+ EW $t$-channel", 3: "+ external photons",
             4: "+ QCD exchange, colour", 5: "+ masses", 6: r"+ $W$+jet, $2\to2$", 7: r"+ $2\to3$",
             8: r"+ $2\to4$", 9: "+ one loop"}
NEW = ["ee_ddbarg", "uubar_Zg", "ee_ttbarg", "ee_ttbar_nlo_thr", "udbar_enu", "ee_dd_ew_nlo"]   # the star arms' probes
S = __import__("cells").S
RUNGS = sorted({r for r in range(1, 10) for n in S if n.startswith(f"tp3_r{r}fte_")})
cols = dict(zip(RUNGS, ps.sequence(len(RUNGS))))


def curve(f):
    D, L = [], []
    for k in range(2, 9):
        v = f(k)
        if v is not None:
            D.append(10 ** (k / 2)); L.append(v)
    return D, L


def draw(ax, f, *a, **kw):
    """One series, left out (and out of the legend) where it has no cell yet."""
    D, L = curve(f)
    if D:
        ax.plot(D, L, *a, **kw)


def fam(r):
    """A pretraining's fine-tune family: rung 0 is the ee->uu pretraining (64k steps, as every rung)."""
    return "tp3_uu64fte" if r == 0 else f"tp3_r{r}fte"


def draw_final(ax, f, p, *a, **kw):
    """One series at the final horizons (cells.final): a line through every cell, a cell still at 8k where the study
    runs 32k drawn open in the series' colour. Returns the cells drawn open."""
    pts = [(10 ** (k / 2),) + final(f, p, k) for k in range(2, 9)]
    pts = [x for x in pts if x[1] is not None]
    if not pts:
        return None
    ax.plot([x[0] for x in pts], [x[1] for x in pts], *a, **kw)
    op = [x for x in pts if not x[2]]
    if op:
        col = kw.get("color")
        ax.plot([x[0] for x in op], [x[1] for x in op], "o", color=col, mfc="white", zorder=5)
    return op


if __name__ == "__main__":
    for part, order in (("ee", P[:6]), ("qcd", P[6:])):
        figs = ps.panels(len(order))
        for (fig, ax), p in zip(figs, order):
            draw_final(ax, "scr", p, "o-", color="k", label="from scratch")
            draw_final(ax, fam(0), p, "o-", color=ps.C.blue, label=r"rung 0")
            for r in RUNGS:
                draw_final(ax, fam(r), p, "o-", color=cols[r], label=f"rung {r}")     # the structures: the text's table
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
            ps.process_label(ax, LAB[p])
            ps.make_room(ax)
        a0 = figs[0][1]
        if USE_32K:
            a0.plot([], [], "o", color="k", mfc="white", label="open: 8k steps where the study runs 32k (for now)")
        ps.shared_legend(figs[0][0], a0, ncol=3)     # eleven series: no panel has a clear corner for them
        ps.save_panels(figs, os.path.join(ROOT, "analysis", "transfer", "figs", f"rung_grid_{part}"))
    # the star arms' probes, from scratch, ee->uu and rung 1 (all at their grid horizons)
    figs = ps.panels(len(NEW))
    for (fig, ax), p in zip(figs, NEW):
        draw(ax, lambda k: scratch(p, k, steered=False)[0], "o-", color="k", label="from scratch")
        draw(ax, lambda k: best(f"{fam(0)}_{p}_d{k}")[0], "o-", color=ps.C.blue, label=r"rung 0: $ee\to u\bar u$")
        draw(ax, lambda k: best(f"{fam(1)}_{p}_d{k}")[0], "o-", color=cols[1], label=f"rung 1: {RUNG_ADDS[1]}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, LAB[p])
        ps.make_room(ax)
    ps.shared_legend(figs[0][0], figs[0][1], ncol=2)
    ps.save_panels(figs, os.path.join(ROOT, "analysis", "transfer", "figs", "rung_grid_new"))
