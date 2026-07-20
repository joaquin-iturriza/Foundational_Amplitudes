#!/usr/bin/env python
"""L2 uugg: ABSOLUTE held-out MSE(Δlog|M|^2) per y_min decade, base (uniform) + sigma-driven at
gamma in {1,2,3,5}. Same view as the left panel of the base-vs-sigma figure, now with all gammas
(markers on the true decade center, no horizontal dodge; mean±seed-spread). png+pdf."""
import os
import numpy as np
import matplotlib.pyplot as plt

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
DEC = [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.01)]
LBL = ["<1e-6", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]


def load(tag):
    per = []
    for s in (0, 1, 2):
        d = np.load(f"{REPO}/analysis/divergences/heldout_eval_{tag}_s{s}.npz")
        e2 = (d["pred_logamp"] - d["true_logamp"]) ** 2; ym = d["y_min"]
        per.append([e2[(ym >= lo) & (ym < hi)].mean() for lo, hi in DEC])
    return np.array(per)


x = np.arange(len(DEC))
series = [("base (uniform)", load("base"), "#444", "o", ":"),
          (r"$\sigma$, $\gamma=1$", load("sigma"), "#f6a300", "s", "-"),
          (r"$\sigma$, $\gamma=2$", load("g2"), "#e85d04", "s", "-"),
          (r"$\sigma$, $\gamma=3$", load("g3"), "#c1121f", "s", "-"),
          (r"$\sigma$, $\gamma=5$", load("g5"), "#6a040f", "s", "-")]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for name, arr, c, mk, ls in series:
    ax.errorbar(x, arr.mean(0), yerr=arr.std(0), fmt=mk, ls=ls, color=c, capsize=2,
                mfc=("white" if name.startswith("base") else c), label=name)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(LBL)
ax.set_xlabel(r"$y_{\min}$ decade (deep IR $\to$ bulk)")
ax.set_ylabel(r"held-out MSE $\Delta\log|\mathcal{M}|^2$")
ax.set_title(r"$ee\to u\bar u gg$ deep-IR error per decade (3 seeds, matched $\mu$/budget)")
ax.legend(); ax.grid(alpha=0.3, which="both")
fig.tight_layout()
base = os.path.join(REPO, "analysis/divergences/figs/l2_uugg_perdecade_abs")
os.makedirs(os.path.dirname(base), exist_ok=True)
fig.savefig(base + ".png", dpi=140); fig.savefig(base + ".pdf")
print(f"saved {base}.png / .pdf")
