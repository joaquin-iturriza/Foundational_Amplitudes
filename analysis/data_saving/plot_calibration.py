"""Best validation MSE(ln|M|^2) vs training-set size D, per target (tree-arm calibration)."""
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import plot_style as ps

DATA = os.path.join(REPO, "analysis/data_saving/data/horizons_cal2.json")
rows = json.load(open(DATA))
targets = [
    ("epem_ttx", r"$e^+e^-\to t\bar t$", ps.C.blue),
    ("epem_ccxg", r"$e^+e^-\to c\bar c g$", ps.C.vermillion),
    ("epem_bbxgg", r"$e^+e^-\to b\bar b gg$", ps.C.green),
]
fig, ax = ps.figure()
for t, lab, col in targets:
    r = sorted((r for r in rows if r["target"] == t), key=lambda r: r["D"])
    ax.plot([x["D"] for x in r], [x["best"] for x in r], marker="o", color=col, label=lab)
ax.axhline(1e-4, color="gray", linestyle="--", label=r"$10^{-4}$")
ax.axhline(1e-2, color="gray", linestyle=":", label=r"$10^{-2}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$D$")
ax.set_ylabel(r"best val MSE$(\ln|\mathcal{M}|^2)$")
ps.legend(ax, "upper right")
ps.save(fig, os.path.join(REPO, "analysis/data_saving/figs/calibration_best_mse"))
