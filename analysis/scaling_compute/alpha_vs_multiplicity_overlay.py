#!/usr/bin/env python3
"""Compute-scaling exponent alpha_C vs final-state multiplicity, styled after the
alphas_panels figure of the Quantifying-ML-uncertainties talk: log-y axis, grey
shaded wedge below the theoretical lower bound alpha = 4/DOF (DOF = 3*n_fs - 4)
with the label rotated along the dashed line, one tab10 colour per process
family joined by dotted lines, fine-tuned (full, layer-decay) targets as stars.

Exponents are read from the two scaling_law_params.json files so the figure
tracks the actual fits; the ratio (virt/Born) targets are excluded as nonsense.
Usage: python alpha_vs_multiplicity_overlay.py [out_basename]
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

ROOT_FOR_STYLE = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, ROOT_FOR_STYLE)
import plot_style as ps  # noqa: E402

ROOT = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
SOLO = os.path.join(ROOT, "sweeps/scaling_solo_full/scaling_law_params.json")
FT   = os.path.join(ROOT, "sweeps/finetune_scaling_virt_002/scaling_law_params.json")
out  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "analysis/scaling_compute/alpha_vs_multiplicity_overlay")

# process families: colour + members as (json key stem, n_fs). tab10 colours as
# in the talk (cZ/cWZ/cAA/cWWZ/cttH).
FAMILIES = [
    (r"$ee\to q\bar q(+ng)$", "#2ca02c",
     [("ee_uu_91-1000GeV", 2), ("ee_uug_91-1000GeV", 3), ("ee_uugg_91-1000GeV", 4)]),
    (r"$ee\to \gamma\gamma(+\gamma)$", "#9467bd",
     [("ee_aa_10-1000GeV", 2), ("ee_aaa_10-1000GeV", 3)]),
    (r"$ee\to WW(+Z)$", "#1f77b4",
     [("ee_WW_162-1000GeV", 2), ("ee_wwz_255-1000GeV", 3)]),
    (r"$ee\to t\bar t$", "#e377c2", [("ee_ttbar_346-1000GeV", 2)]),
    (r"$ee\to t\bar t$ virt", "#d62728", [("ee_ttbar_nlo_virt_e4", 2)]),
    (r"$ee\to q\bar q$ virt", "#ff7f0e", [("ee_uu_nlo_virt_e4", 2)]),
]
# fine-tuned (starred) points: same colour as the matching from-scratch family
FT_STARS = [("ee_ttbar_nlo_virt_e4", 2, "#d62728"),
            ("ee_uu_nlo_virt_e4", 2, "#ff7f0e")]

def load(path):
    try:
        d = json.load(open(path))
    except Exception as e:
        print("WARN could not read", path, e); return {}
    def norm(k):
        return k[:-len("_amplitudes")] if k.endswith("_amplitudes") else k
    return {norm(k): float(v["alpha"]) for k, v in d.items()
            if isinstance(v, dict) and v.get("alpha") is not None
            and "ratio" not in norm(k)}

solo, ft = load(SOLO), load(FT)

# The per-family colours are deliberately the talk's tab10 assignment, not the shared
# qualitative palette: they identify process families consistently with the reference figure.
fig, ax = ps.figure()
ax.set_xscale("log"); ax.set_yscale("log")

# As in the talk figure: x is the phase-space DOF = 3 n_fs - 4 on a log axis
# (ticks labelled by the particle count n_fs), so the theoretical lower bound
# alpha = 4/DOF is an exactly straight slope -1 line, with a grey shaded wedge
# below and the label rotated along it.
dofx = lambda n: 3.0 * n - 4.0
XLIM = (1.55, 10.5)
YLIM = (4.0 / XLIM[1], 4.0 / XLIM[0])  # bound runs corner to corner
dd = np.array(XLIM)
ax.plot(dd, 4.0 / dd, color="0.65", ls="--", zorder=1)
ax.fill_between(dd, YLIM[0], 4.0 / dd, color="0.5", alpha=0.14, zorder=0, lw=0)
# The bound is labelled ALONG THE LINE, rotated to its slope, as in the reference figure. This
# is the sanctioned second in-axes label: it names a line that a legend entry would only put at
# one remove from the thing it names, and here the line IS the figure's reference.
x0 = np.sqrt(XLIM[0] * XLIM[1])
p = lambda x: ax.transData.transform((x, 4.0 / x))
(dx, dy) = p(x0 * 1.3) - p(x0 / 1.3)
ax.text(x0, (4.0 / x0) * 0.66, "Theoretical lower bound", color="0.55",
        rotation=np.degrees(np.arctan2(dy, dx)),
        rotation_mode="anchor", ha="center", va="center", zorder=1)

# small multiplicative jitter so overlapping n_fs=2 points separate
jit = {2: np.geomspace(1 / 1.06, 1.06, sum(n == 2 for _, _, m in FAMILIES for _, n in m))}
taken = {2: 0}
def jx(n):
    if n not in jit: return dofx(n)
    x = dofx(n) * jit[n][taken[n]]; taken[n] += 1; return x

for lab, color, members in FAMILIES:
    pts = [(jx(n), solo[k]) for k, n in members if k in solo]
    if not pts: continue
    xs, ys = zip(*pts)
    ax.plot(xs, ys, ls=":", lw=1.6, color=color, zorder=2)
    ax.scatter(xs, ys, s=45, color=color, zorder=3, label=lab)

for k, n, color in FT_STARS:
    if k in ft:
        ax.scatter([dofx(n) * 1.10], [ft[k]], s=90, marker="*", color=color,
                   zorder=4, label=None)
ax.scatter([], [], s=90, marker="*", color="0.3", label="fine-tuned")

ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
ax.xaxis.set_major_locator(FixedLocator([dofx(n) for n in (2, 3, 4)]))
ax.xaxis.set_minor_locator(FixedLocator([]))
ax.set_xticklabels(["2", "3", "4"])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2]))
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("final-state particles")
ax.set_ylabel(r"$\alpha_C$")
# Eight entries with long process labels cannot go inside a 2.40in plot box at 11pt: the
# legend was covering most of the panel and printing over the fine-tuned stars. This is the
# sanctioned last resort -- one strip above the plot, paid for by the CANVAS, so the plot box
# is still the standard one.
ps.shared_legend(fig, ax, ncol=2)
ax.grid(False)
ps.save(fig, out)
print("saved", out + ".png", out + ".pdf",
      "| solo pts:", sum(k in solo for _, _, m in FAMILIES for k, _ in m),
      "ft stars:", sum(k in ft for k, _, _ in FT_STARS))
