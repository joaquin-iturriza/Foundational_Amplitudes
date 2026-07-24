#!/usr/bin/env python
"""How much data saturates a region? -- the sigma-tail decay vs pool size.

The L2 loop logs, every round, the sigma distribution over the FRESH proposal batch (drawn from the
same fixed base every round, so it measures the model's remaining uncertainty over the whole space,
comparable across rounds). The p99 -- the hard tail, i.e. the singular region the bulk has long since
left behind -- is the stopping statistic: when it stops falling as the pool grows, that region has
absorbed all the data it can at this model size / horizon, and adding more is waste.

Panel (a): sigma p99 vs POOL SIZE for several n_total budgets. The end-of-training floor is
essentially flat in the data budget -- base 0.0448/0.0468/0.0477 and sigma 0.0420/0.0418/0.0427 at
N=75k/150k/600k -- i.e. 8x more generated data does NOT lower it (for the base arm it is marginally
WORSE, +6.4%). IMPORTANT CONFOUND: every run uses the same 4000 steps, so a larger pool gets
proportionally less optimization per event (~870 passes at 75k vs ~109 at 600k). The defensible claim
is therefore "at FIXED optimizer budget, more unique data buys no reduction in residual uncertainty",
i.e. the binding constraint here is capacity/optimization rather than coverage -- NOT that 75k events
is all this region could ever absorb. Separating the two needs a fixed-EPOCH (steps scaled with data)
or capacity sweep. Note also the sigma arm floors ~8% below base at every budget.
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
    """Load every satlog for `arm`. Raises if none are found: a missing satlogs dir must NOT
    silently yield an empty figure -- that once shipped a blank plot into the paper build after the
    logs were deleted with a worktree. Fail loudly instead."""
    files = sorted(glob.glob(os.path.join(HERE, f"satlogs/uugg_{arm}_n*.json")))
    if not files:
        raise SystemExit(
            f"[plot_saturation] no satlogs matching satlogs/uugg_{arm}_n*.json under {HERE}.\n"
            f"  These are written by l2_online_uugg.py --sat_log and are gitignored (regenerable).\n"
            f"  Re-run: PROCESS=uugg ARM={arm} NTOTAL=<N> TAG=sat<N> "
            f"SAT_LOG=analysis/divergences/satlogs/uugg_{arm}_n<N>.json sbatch analysis/divergences/run_l2_proc.sh")
    out = {}
    for f in files:
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
ax.axhspan(0.041, 0.049, color="k", alpha=0.06)
ax.text(ax.get_xlim()[0]*1.1, 0.045, "floor 0.042-0.048\n(8x data changes it by <7%)", fontsize=8, va="center")
ax.set_title("(a) 8$\\times$ more data does not lower the $\\sigma$-tail floor", fontsize=10.5)
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

fig.suptitle(r"Saturation: at fixed optimizer budget the uncertainty tail floors regardless of data volume", fontsize=12)
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
