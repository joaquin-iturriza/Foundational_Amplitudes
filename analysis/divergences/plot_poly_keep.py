#!/usr/bin/env python
"""Polynomial keep rule vs the sigma^gamma power law: does curvature buy anything?

The keep rule log w = c1*u + c2*u^2 + c3*u^3 (u = log sigma - median) nests the power law at c2=c3=0.
A 12-trial single-fidelity DyHPO sweep on uugg (objective = logflat MSE) explores it. This plots the
sweep outcome against the two things that decided it:

Left  : logflat MSE vs the quadratic coefficient c2, coloured by c1 (=gamma). The power-law family is
        the c2=0 line; the E[err^2|sigma] pre-test predicted POSITIVE c2 should help. The best two
        configs both sit at c2 ~ +1.3..1.4, and every negative-c2 config is mid-pack or worse.
Right : the best polynomial (3-seed) against the best power-law arm sigma^10 (3-seed) -- the honest
        comparison, since the single-seed 3% edge is inside seed scatter.

Reads sweeps/l2_poly_uugg/results/*.json for the sweep, and heldout_eval_{polyopt,g10}_s{0,1,2}.npz
for the 3-seed confirmation (the latter written by the confirmation runs). Emits png+pdf.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEC = [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.01)]


def logflat(tag):
    d = np.load(os.path.join(HERE, tag + ".npz"))
    e2 = (d["pred_logamp"] - d["true_logamp"]) ** 2
    y = d["y_min"]
    return np.mean([e2[(y >= lo) & (y < hi)].mean() for lo, hi in DEC if ((y >= lo) & (y < hi)).sum() > 0])


# --- sweep results ---
rows = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "..", "..", "sweeps",
        "l2_poly_uugg", "results", "*.json"))]
if not rows:  # fall back to the worktree-local sweeps dir layout
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "..", "sweeps",
            "l2_poly_uugg", "results", "*.json"))]
c1 = np.array([r["keep_c1"] for r in rows]); c2 = np.array([r["keep_c2"] for r in rows])
lf = np.array([r["logflat_mse"] for r in rows])
POWER_BEST_1SEED = 3.351e-2         # sigma^10 seed 0

fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))

# ---------------------------------------------------------------- (a) logflat vs c2
ax = axes[0]
sc = ax.scatter(c2, lf, c=c1, s=90, cmap="viridis", edgecolor="k", zorder=3)
plt.colorbar(sc, ax=ax, label=r"$c_1$  (= $\gamma$, linear term)")
ax.axvline(0.0, color="crimson", ls="--", lw=1.5)
ax.text(0.05, ax.get_ylim()[1], "power-law family\n($c_2{=}c_3{=}0$)", color="crimson",
        fontsize=8, va="top")
ax.axhline(POWER_BEST_1SEED, color="0.4", ls=":", lw=1.4)
ax.text(ax.get_xlim()[1], POWER_BEST_1SEED, r"best $\sigma^\gamma$ (seed 0)", fontsize=8,
        ha="right", va="bottom", color="0.4")
# mark the winner
i0 = int(np.argmin(lf))
ax.annotate("best polynomial\n" fr"$c_2={c2[i0]:+.2f}$", xy=(c2[i0], lf[i0]),
            xytext=(10, 18), textcoords="offset points", fontsize=8.5,
            arrowprops=dict(arrowstyle="->", lw=1))
ax.set_xlabel(r"quadratic coefficient $c_2$  (curvature in $\log\sigma$)")
ax.set_ylabel("held-out logflat MSE")
ax.set_title(r"(a) positive curvature helps; the power law ($c_2{=}0$) is nested", fontsize=10)
ax.grid(True, alpha=0.25)

# ---------------------------------------------------------------- (b) 3-seed confirmation
ax = axes[1]
poly = [logflat(f"heldout_eval_uugg_polyopt_s{s}") for s in (0, 1, 2)
        if os.path.exists(os.path.join(HERE, f"heldout_eval_uugg_polyopt_s{s}.npz"))]
powr = [logflat(f"heldout_eval_g10_s{s}") for s in (0, 1, 2)
        if os.path.exists(os.path.join(HERE, f"heldout_eval_g10_s{s}.npz"))]
labels, groups, colors = [], [], []
if powr:
    labels.append(r"power law $\sigma^{10}$" "\n(3 seeds)"); groups.append(powr); colors.append("0.5")
if poly:
    labels.append(r"polynomial $c_2{=}1.3$" "\n(3 seeds)"); groups.append(poly); colors.append("crimson")
if groups:
    xs = np.arange(len(groups))
    means = [np.mean(g) for g in groups]; stds = [np.std(g) for g in groups]
    ax.bar(xs, means, 0.55, yerr=stds, color=colors, capsize=6, alpha=0.85)
    for x, g in zip(xs, groups):
        ax.plot([x] * len(g), g, "o", color="k", ms=6, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("held-out logflat MSE")
    lo = min(min(g) for g in groups) * 0.97; hi = max(max(g) for g in groups) * 1.03
    ax.set_ylim(lo, hi)
    if len(groups) == 2:
        ax.set_title(f"(b) 3-seed: poly {means[-1]:.3e} vs power {means[0]:.3e}", fontsize=10)
else:
    ax.text(0.5, 0.5, "3-seed confirmation pending", ha="center", va="center",
            transform=ax.transAxes, fontsize=11)
    ax.set_title("(b) 3-seed confirmation", fontsize=10)
ax.grid(True, axis="y", alpha=0.25)

fig.suptitle(r"Polynomial keep rule: positive curvature $c_2$ is the helpful direction, "
             r"but the gain over well-tuned $\sigma^\gamma$ is within seed scatter", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
base = os.path.join(HERE, "figs", "l2_poly_keep")
os.makedirs(os.path.dirname(base), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{base}.{ext}", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {base}.png/.pdf")
if groups and len(groups) == 2:
    print(f"  power sigma^10: {np.mean(powr):.4e} ± {np.std(powr):.4e}")
    print(f"  polynomial    : {np.mean(poly):.4e} ± {np.std(poly):.4e}")
