#!/usr/bin/env python
"""How much data saturates a region? -- the sigma-tail decay vs pool size.

The L2 loop logs, every round, the sigma distribution over the FRESH proposal batch (drawn from the
same fixed base every round, so it measures the model's remaining uncertainty over the whole space,
comparable across rounds). The p99 -- the hard tail, i.e. the singular region the bulk has long since
left behind -- is the stopping statistic: when it stops falling as the pool grows, that region has
absorbed all the data it can at this model size / horizon, and adding more is waste.

Panel (a): sigma p99 vs POOL SIZE for several n_total budgets. The curves do NOT collapse on pool
(a larger budget takes fewer optimizer steps to reach a given pool, so its live sigma sits higher
there); the point is that all budgets converge to the SAME tail floor (~0.046) by end of training,
and 8x more data does not lower it -- the deep region is saturated by ~75k events at this model size
and horizon, so more data there is waste.
Panel (b): the per-round fractional fall in p99. Crossing below the tolerance band (and staying there
for `patience` rounds) is the machine-checkable stopping rule the --stop_on_saturation latch uses.

Reads analysis/divergences/satlogs/uugg_<arm>_n<N>.json. Emits png+pdf.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SAT_TOL = 0.02          # matches the driver default --sat_tol

# n_total -> colour; base solid, sigma dashed
COLORS = {75000: "steelblue", 150000: "darkorange", 300000: "seagreen", 600000: "crimson"}


def load(arm):
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, f"satlogs/uugg_{arm}_n*.json"))):
        d = json.load(open(f))
        rr = d["rounds"]
        out[d["n_total"]] = dict(
            pool=np.array([r["pool"] for r in rr]),
            p99=np.array([r["sigma_p99"] for r in rr]),
            p90=np.array([r["sigma_p90"] for r in rr]),
            p50=np.array([r["sigma_p50"] for r in rr]),
            drop=np.array([np.nan if r["rel_drop_p99"] is None else r["rel_drop_p99"] for r in rr]),
        )
    return out


fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))

# ---------------------------------------------------------------- (a) sigma tail vs pool size
ax = axes[0]
for arm, ls, mk in [("base", "-", "o"), ("sigma", "--", "s")]:
    data = load(arm)
    for N, d in sorted(data.items()):
        c = COLORS.get(N, "gray")
        ax.plot(d["pool"], d["p99"], ls + mk, color=c, lw=1.8, ms=5,
                label=f"{arm}, N={N//1000}k")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("training pool size (events)")
ax.set_ylabel(r"$\sigma$ tail (p99) over the fresh proposal batch")
ax.axhspan(0.042, 0.050, color="k", alpha=0.06)
ax.text(ax.get_xlim()[0]*1.1, 0.046, "common floor ~0.046", fontsize=8, va="center")
ax.set_title("(a) every budget converges to the same $\\sigma$-tail floor", fontsize=10.5)
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=7.5, ncol=2, loc="upper right")

# ---------------------------------------------------------------- (b) per-round fractional fall
ax = axes[1]
base = load("base")
for N, d in sorted(base.items()):
    c = COLORS.get(N, "gray")
    ax.plot(d["pool"], d["drop"], "-o", color=c, lw=1.8, ms=5, label=f"N={N//1000}k")
ax.axhspan(0, SAT_TOL, color="k", alpha=0.08)
ax.axhline(SAT_TOL, color="k", ls=":", lw=1.2)
ax.text(ax.get_xlim()[1], SAT_TOL * 1.15, f"saturation tol = {SAT_TOL:.0%}", fontsize=8,
        ha="right", va="bottom")
ax.set_xscale("log")
ax.set_xlabel("training pool size (events)")
ax.set_ylabel(r"per-round fractional fall in $\sigma$ p99")
ax.set_title("(b) the stopping rule: fall drops below tolerance", fontsize=10.5)
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=8, loc="upper right")

fig.suptitle(r"Saturation: the uncertainty tail converges to a data-independent floor -- more data past "
             r"$\sim$75k does not lower it", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
base_path = os.path.join(HERE, "figs", "l2_saturation")
os.makedirs(os.path.dirname(base_path), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{base_path}.{ext}", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {base_path}.png/.pdf")

# text summary
for arm in ("base", "sigma"):
    for N, d in sorted(load(arm).items()):
        floor = d["p99"][-1]
        print(f"  {arm:5s} N={N//1000:>3d}k: p99 {d['p99'][0]:.4f} -> {floor:.4f}  "
              f"(final fall {100*d['drop'][-1]:.1f}%, pool {d['pool'][-1]})")
