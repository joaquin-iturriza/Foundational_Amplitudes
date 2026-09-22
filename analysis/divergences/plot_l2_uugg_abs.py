#!/usr/bin/env python
"""Steering uugg: ABSOLUTE held-out MSE(Δlog|M|^2) per y_min decade, base (uniform) + sigma-driven at
gamma in {1,2,3,5,10}. Same view as the left panel of the base-vs-sigma figure, now with all gammas
(markers on the true decade center, no horizontal dodge; mean±seed-spread). png+pdf."""
import sys

import numpy as np

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

DEC = [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.01)]
LBL = [r"$<\!-6$", r"$-6$", r"$-5$", r"$-4$", r"$-3$", r"$-2$", r"$-1$"]


def load(tag):
    per = []
    for s in (0, 1, 2):
        d = np.load(f"{REPO}/analysis/divergences/heldout_eval_{tag}_s{s}.npz")
        e2 = (d["pred_logamp"] - d["true_logamp"]) ** 2; ym = d["y_min"]
        per.append([e2[(ym >= lo) & (ym < hi)].mean() for lo, hi in DEC])
    return np.array(per)


x = np.arange(len(DEC))
GAMMAS = [1, 2, 3, 5, 10]
TAGS = {1: "sigma", 2: "g2", 3: "g3", 5: "g5", 10: "g10"}
ramp = ps.sequence(len(GAMMAS))

fig, ax = ps.figure()
b = load("base")
ax.errorbar(x, b.mean(0), yerr=b.std(0), fmt="o:", color=ps.C.grey, mfc="white", label="uniform")
for g, c in zip(GAMMAS, ramp):
    arr = load(TAGS[g])
    ax.errorbar(x, arr.mean(0), yerr=arr.std(0), fmt="s-", color=c, label=rf"$\sigma$, $\gamma={g}$")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(LBL)
ax.set_xlabel(r"$\log_{10} y_{\min}$")
ax.set_ylabel(r"MSE$(\Delta\log|\mathcal{M}|^2)$")
ax.legend(ncol=2, loc="lower left")
ps.process_label(ax, r"$e^+e^-\to u\bar u gg$", loc="upper right")
ps.save(fig, "analysis/divergences/figs/l2_uugg_perdecade_abs")
