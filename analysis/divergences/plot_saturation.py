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
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

REPO = "/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAT_TOL = 0.02          # matches the driver default --sat_tol

# n_total -> colour along an ordered ramp; base solid, sigma dashed
_NS = [75000, 150000, 300000, 600000]
COLORS = dict(zip(_NS, ps.sequence(len(_NS))))


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


fig, axes = ps.figure(ncols=2)

# ---------------------------------------------------------------- (a) sigma tail vs pool size
ax = axes[0]
for arm, ls, mk in [("base", "-", "o"), ("sigma", "--", "s")]:
    data = load(arm)
    for N, d in sorted(data.items()):
        c = COLORS.get(N, ps.C.grey)
        ax.plot(d["pool"], d["p99"], ls + mk, color=c, label=f"{arm} {N//1000}k")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("training pool size")
ax.set_ylabel(r"$\sigma$ p99")
# All six curves named outright -- three pool sizes x two arms. Factorising this into three
# colour entries plus a "solid means base, dashed means sigma" key saved almost no width and
# made the reader assemble each curve's identity themselves. ps.legend drops the columns first
# and then the font (to LEGEND_MIN_PT) until the full six fit inside the plot box.
ps.legend(ax, "lower left", ncol=2, columnspacing=1.0)
ps.process_label(ax, r"$e^+e^-\to u\bar u gg$", loc="upper right")

# ---------------------------------------------------------------- (b) per-round fractional fall
ax = axes[1]
base = load("base")
ax.axhline(SAT_TOL, color=ps.C.grey, ls=":", label=rf"tolerance ${100*SAT_TOL:.0f}\%$")
for N, d in sorted(base.items()):
    c = COLORS.get(N, ps.C.grey)
    ax.plot(d["pool"], d["drop"], "-o", color=c, label=f"{N//1000}k")
ax.set_xscale("log")
ax.set_xlabel("training pool size")
ax.set_ylabel(r"fractional fall in $\sigma$ p99")
ax.legend(loc="upper right")

base_path = os.path.join(HERE, "figs", "l2_saturation")
ps.save(fig, base_path)
print(f"wrote {base_path}.png/.pdf")

# text summary
for arm in ("base", "sigma"):
    for N, d in sorted(load(arm).items()):
        floor = d["p99"][-1]
        print(f"  {arm:5s} N={N//1000:>3d}k: p99 {d['p99'][0]:.4f} -> {floor:.4f}  "
              f"(final fall {100*d['drop'][-1]:.1f}%, pool {d['pool'][-1]})")
