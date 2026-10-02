"""Transfer study, the fine-tune grid: each probe's loss against D, from scratch and fine-tuned from each pretraining
in the grid's setup (tp3_<parent>fte: training.lr searched over [1e-3, 1e-2], lr_scale = layer_decay = 1, 5 trials;
docs/results.tex sec:ladder), with the ee->uu fine-tune of Table tab:ladder_transfer_all (cells.py, the earlier
setup) for reference. A cell's value is its search's best so far (MSE of log|M|^2 at the best checkpoint); cells still
running are drawn as they stand. ee->WW on the mixture pool throughout.
  <base>_ee_a..f, <base>_qcd_a..f
    python analysis/transfer/rung_grid.py      -> figs/rung_grid
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, ARM, best, scratch, finetune  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

LAB = {"ee_ddbar": r"$e^+e^-\to d\bar d$", "ee_nnbar": r"$e^+e^-\to\nu_e\bar\nu_e$", "ee_ttbar": r"$e^+e^-\to t\bar t$",
       "ee_WW": r"$e^+e^-\to W^+W^-$", "ee_dd_nlo": r"$e^+e^-\to d\bar d$ (1-loop)",
       "ee_bb_nlo": r"$e^+e^-\to b\bar b$ (1-loop)", "ee_Za": r"$e^+e^-\to Z\gamma$", "ud_ud": r"$ud\to ud$",
       "uubar_gg": r"$u\bar u\to gg$", "uubar_Zg": r"$u\bar u\to Zg$", "uubar_Zgg": r"$u\bar u\to Zgg$",
       "uubar_Zggg": r"$u\bar u\to Zggg$"}
P = ["ee_ddbar", "ee_nnbar", "ee_ttbar", "ee_WW", "ee_dd_nlo", "ee_bb_nlo", "ee_Za", "ud_ud", "uubar_gg",
     "uubar_Zg", "uubar_Zgg", "uubar_Zggg"]
RUNGS = [r for r in range(1, 10) if any(n.startswith(f"tp3_r{r}fte_") for n in __import__("cells").S)]
cols = ps.sequence(len(RUNGS))


def curve(f):
    D, L = [], []
    for k in range(2, 9):
        v = f(k)
        if v is not None:
            D.append(10 ** (k / 2)); L.append(v)
    return D, L


for part, order in (("ee", P[:6]), ("qcd", P[6:])):
    figs = ps.panels(len(order))
    for (fig, ax), p in zip(figs, order):
        ax.plot(*curve(lambda k: scratch(p, k, steered=False)[0]), "o-", color="k", label="from scratch")
        ax.plot(*curve(lambda k: finetune(p, k, steered=False)[0]), "o--", color=ps.C.blue, label=r"$ee\to u\bar u$, earlier setup")
        ax.plot(*curve(lambda k: best(f"tp3_uufte_{p}_d{k}")[0]), "s-", color=ps.C.blue, mfc="none", label=r"$ee\to u\bar u$")
        for r, c in zip(RUNGS, cols):
            ax.plot(*curve(lambda k: best(f"tp3_r{r}fte_{p}_d{k}")[0]), "o-", color=c, label=f"rung {r}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, LAB[p])
        ps.make_room(ax)
    ps.shared_legend(figs[0][0], figs[0][1], ncol=2)     # seven series: no panel has a clear corner for them
    ps.save_panels(figs, os.path.join(ROOT, "analysis", "transfer", "figs", f"rung_grid_{part}"))
