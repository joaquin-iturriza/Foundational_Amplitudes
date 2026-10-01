"""Transfer study, chapter one: ee->dd~ (near) and uu~->gg (far), from scratch and fine-tuned from the factor-off
ee->uu pretraining, each cell's value by the rule of cells.py (one search per cell, extended in place where its best
sat at the top of its lr window).
  <base>_a  ee->dd~        <base>_b  uu~->gg
    python analysis/transfer/transfer_ch1.py
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "analysis", "transfer", "figs")
sys.path.insert(0, ROOT)
import plot_style as ps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import scratch, finetune  # noqa: E402

figs = ps.panels(2)
for (fig, ax), (p, lab) in zip(figs, (("ee_ddbar", r"$e^+e^-\to d\bar d$"), ("uubar_gg", r"$u\bar u\to gg$"))):
    for arm, col, name in ((scratch, ps.C.blue, "from scratch"), (finetune, ps.C.vermillion, "fine-tuned")):
        pts = [(10 ** (k / 2), arm(p, k)[0]) for k in range(2, 9)]
        pts = [(d, l) for d, l in pts if l is not None]
        ax.plot(*zip(*pts), "o-", color=col, label=name)
        print(p, name, " ".join(f"{d:.0f}:{l:.2g}" for d, l in pts))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(ax, lab)
    ps.legend(ax, "lower left")
ps.save_panels(figs, os.path.join(FIG, "transfer_ch1"))
