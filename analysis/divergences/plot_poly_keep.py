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
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

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
if not rows:
    # An empty sweep dir must NOT yield a silently blank panel -- that is exactly how the
    # missing left panel shipped into the document. sweeps/l2_poly_uugg was destroyed with
    # a worktree (see scripts/fold_worktree.sh) and no copy survives anywhere on disk.
    raise SystemExit(
        "[plot_poly_keep] no trials found under sweeps/l2_poly_uugg/results/.\n"
        "  That 12-trial DyHPO sweep was lost when its worktree was removed without\n"
        "  folding results back. The left panel cannot be rebuilt without re-running it:\n"
        "    python sweep/generate_sweep.py --config sweep/l2_poly_sweep_config.yaml\n"
        "  The right panel (3-seed confirmation) does not depend on the sweep.")
c1 = np.array([r["keep_c1"] for r in rows]); c2 = np.array([r["keep_c2"] for r in rows])
lf = np.array([r["logflat_mse"] for r in rows])
POWER_BEST_1SEED = 3.351e-2         # sigma^10 seed 0

fig, axes = ps.figure(ncols=2)

# ---------------------------------------------------------------- (a) logflat vs c2
ax = axes[0]
sc = ax.scatter(c2, lf, c=c1, s=45, cmap=ps.CMAP, zorder=3)
# Colourbar in an INSET above the panel, not `colorbar(ax=ax)`. Attaching it to the axes takes
# its width out of that gridspec column only, so the left panel came out 1.75in against the
# right one's 2.19in -- two different panel sizes inside one figure, which is the most visible
# form of the inconsistency. An inset is laid out in axes coordinates, so both panels keep the
# same box and the bar is paid for out of the margin (which _expand_to_content then covers).
cax = ax.inset_axes([0.0, 1.04, 1.0, 0.05])
cax.set_label("<colorbar>")
cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
cax.xaxis.set_ticks_position("top")
cb.set_label(r"$c_1$", labelpad=-38)
ax.axvline(0.0, color=ps.C.vermillion, ls="--", label=r"power law, $c_2{=}c_3{=}0$")
ax.axhline(POWER_BEST_1SEED, color=ps.C.grey, ls=":", label=r"best $\sigma^\gamma$, seed 0")
ax.set_xlabel(r"$c_2$")
ax.set_ylabel("held-out logflat MSE")
ax.legend(loc="upper left")

# ---------------------------------------------------------------- (b) 3-seed confirmation
ax = axes[1]
poly = [logflat(f"heldout_eval_uugg_polyopt_s{s}") for s in (0, 1, 2)
        if os.path.exists(os.path.join(HERE, f"heldout_eval_uugg_polyopt_s{s}.npz"))]
powr = [logflat(f"heldout_eval_g10_s{s}") for s in (0, 1, 2)
        if os.path.exists(os.path.join(HERE, f"heldout_eval_g10_s{s}.npz"))]
labels, groups, colors = [], [], []
if powr:
    labels.append(r"$\sigma^{10}$"); groups.append(powr); colors.append(ps.C.grey)
if poly:
    labels.append(r"$c_2{=}1.3$"); groups.append(poly); colors.append(ps.C.vermillion)
if groups:
    xs = np.arange(len(groups))
    means = [np.mean(g) for g in groups]; stds = [np.std(g) for g in groups]
    ax.bar(xs, means, 0.55, yerr=stds, color=colors, capsize=4)
    for x, g in zip(xs, groups):
        ax.plot([x] * len(g), g, "o", color="k", ms=4, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("held-out logflat MSE")
    lo = min(min(g) for g in groups) * 0.97; hi = max(max(g) for g in groups) * 1.03
    ax.set_ylim(lo, hi)
else:
    ax.set_xticks([])
ax.grid(True, axis="y")
ps.process_label(ax, r"$e^+e^-\to u\bar u gg$", loc="upper right")

base = os.path.join(HERE, "figs", "l2_poly_keep")
ps.save(fig, base)
if groups and len(groups) == 2:
    print(f"  power sigma^10: {np.mean(powr):.4e} ± {np.std(powr):.4e}")
    print(f"  polynomial    : {np.mean(poly):.4e} ± {np.std(poly):.4e}")
