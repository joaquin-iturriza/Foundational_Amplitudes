"""Transfer study, preliminary ladder fine-tunes: each probe fine-tuned from the ee->uu pretraining and from ladder rungs
4 and 9 (their best pretraining trial so far, hp15), at D = 10, 1e2, 1e3, 1e4, against the same scratch cells.
The gain L_scratch / L_fine-tune per cell (cells.py: scratch and the ee->uu fine-tune by the cell rule, one search per
cell extended where its best sat at the lr window's top; the rung fine-tunes are single 8-trial searches, not extended),
ee->WW on the mixture pool throughout; the marker is the geometric mean over the four D, the bar the range. (a) the ee probes, (b) the photon and QCD probes; a superscript (1): one loop.
    python analysis/transfer/ladder_prelim.py      -> figs/ladder_prelim
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, best, scratch, finetune  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

P = ["ee_ddbar", "ee_nnbar", "ee_ttbar", "ee_WW", "ee_dd_nlo", "ee_bb_nlo", "ee_Za", "ud_ud", "uubar_gg",
     "uubar_Zg", "uubar_Zgg", "uubar_Zggg"]
TICK = {"ee_ddbar": r"$d\bar d$", "ee_nnbar": r"$\nu\bar\nu$", "ee_ttbar": r"$t\bar t$", "ee_WW": r"$WW$",
        "ee_dd_nlo": r"$d\bar d^{(1)}$", "ee_bb_nlo": r"$b\bar b^{(1)}$", "ee_Za": r"$Z\gamma$", "ud_ud": r"$ud$",
        "uubar_gg": r"$gg$", "uubar_Zg": r"$Zg$", "uubar_Zgg": r"$Zgg$", "uubar_Zggg": r"$Zggg$"}
KS = (2, 4, 6, 8)


def gains(p, arm):
    out = []
    for k in KS:
        s = scratch(p, k, steered=False)[0]
        if arm == "uu":
            f = finetune(p, k, steered=False)[0]
        else:
            f = best(f"tp3_{arm}ftp_{p}_d{k}")[0]
        out.append(s / f)
    return np.array(out)


GROUPS = [P[:6], P[6:]]
fig, axes = ps.figure(ncols=2)
for ax, group in zip(axes, GROUPS):
    x = np.arange(len(group))
    for j, (arm, col, lab) in enumerate((("uu", ps.C.blue, r"$ee\to u\bar u$"), ("r4", ps.C.green, "rung 4"),
                                          ("r9", ps.C.vermillion, "rung 9"))):
        G = [gains(p, arm) for p in group]
        gm = np.array([np.exp(np.mean(np.log(g))) for g in G])
        lo, hi = np.array([g.min() for g in G]), np.array([g.max() for g in G])
        ax.errorbar(x + (j - 1) * 0.25, gm, yerr=[gm - lo, hi - gm], fmt="o", color=col, label=lab)
        print(lab, " ".join(f"{p}:{v:.2f}" for p, v in zip(group, gm)))
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([TICK[p] for p in group])
    ax.set_ylabel(r"$L_{\rm scratch}/L_{\rm fine\text{-}tune}$")
    ax.set_ylim(0.05, 300)
ps.legend(axes[0], "upper left")
ps.save(fig, os.path.join(ROOT, "analysis", "transfer", "figs", "ladder_prelim"))
