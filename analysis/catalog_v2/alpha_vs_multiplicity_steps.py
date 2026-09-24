"""Scaling exponent against final-state multiplicity for the catalog steps curve: the same figure as
analysis/scaling_compute/alpha_vs_multiplicity_overlay.py (fig:alphamult) -- log alpha against the
phase-space DOF = 3 n_fs - 4 on a log axis (ticks by n_fs), the bound alpha = 4/DOF corner to corner
with a shaded wedge below and its label along the line, one colour per family joined by
dotted lines -- for the families of that figure as they appear in catalog_v2.
Dots: the process in the joint run (full pools, 1000-8000 steps x three seeds; steps_tuned.json),
fitted on its own points with the floor-aware law A C^-alpha + L_inf. Open squares: the same process
trained alone (bs 1024, full pool, sweeps/solob1k_t*), where there is such a reference. One panel
per training aggregation: (a) arithmetic mean, (b) geometric mean.
    python analysis/catalog_v2/alpha_vs_multiplicity_steps.py
Writes analysis/catalog_v2/alpha_vs_multiplicity_steps (png + pdf)."""
import json, os, sys
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "sweep"))
import plot_style as ps
from analyze_pretraining_scaling import fit_power_law_with_floor, flops_per_step

D = json.load(open(os.path.join(HERE, "steps_tuned.json")))
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
FIT_STEPS, SOLO_STEPS = [1000, 2000, 4000, 8000], [33, 67, 134, 268, 536, 1072]
runs = [r for r in D["joint"] if np.median(list(r["final"].values())) <= 0.5 and r["steps"] in FIT_STEPS]
nbar = float(np.mean([NP[n] for n in {n for r in runs for n in r["final"]} if n in NP]))
# the families of fig:alphamult, same tab10 colours, followed through catalog_v2's multiplicities
FAMILIES = [
    (r"$ee\to q\bar q(+ng)$", "#2ca02c", ["ee_uu", "ee_uug", "ee_uugg"]),
    (r"$ee\to\gamma\gamma(+n\gamma)$", "#9467bd", ["ee_aa", "ee_aaa", "ee_aaaa"]),
    (r"$ee\to WW(+Z,ZZ)$", "#1f77b4", ["ee_WW", "ee_wwz", "ee_WWZZ"]),
    (r"$ee\to t\bar t(+ng)$", "#e377c2", ["ee_ttbar", "ee_ttbarg", "ee_ttbargg"]),
    (r"$ee\to t\bar t(+g)$, 1-loop", "#d62728", ["ee_ttbar_nlo", "ee_ttbarg_nlo"]),
    (r"$ee\to q\bar q(+g)$, 1-loop", "#ff7f0e", ["ee_uu_nlo", "ee_uug_nlo"]),
]

def alpha(c, l):
    f = fit_power_law_with_floor(c, l) if len(c) >= 4 else None
    return None if f is None else f[1]

def joint_alpha(arm, n):
    pts = [(flops_per_step(8, nbar, 16384) * r["steps"], r["final"][n]) for r in runs if r["arm"] == arm and n in r["final"]]
    return alpha([p[0] for p in pts], [p[1] for p in pts])

def solo_alpha(n):
    t = [s for s in SOLO_STEPS if f"{n}|{s}" in D["solo"]]
    return alpha([flops_per_step(8, NP[n], 1024) * s for s in t], [D["solo"][f"{n}|{s}"] for s in t]) if t else None

dofx = lambda n: 3.0 * n - 4.0
XLIM = (1.55, 10.5); YLIM = (4.0 / XLIM[1], 4.0 / XLIM[0])     # the bound runs corner to corner
fig, axes = ps.figure(ncols=2, sharey=True)      # the same quantity in both panels: one scale
for ax, arm in zip(axes, ("arith", "geo")):
    ax.set_xscale("log"); ax.set_yscale("log")
    dd = np.array(XLIM)
    ax.plot(dd, 4.0 / dd, color="0.65", ls="--", zorder=1)
    ax.fill_between(dd, 1e-3, 4.0 / dd, color="0.5", alpha=0.14, zorder=0, lw=0)
    print(f"== {arm}")
    for j, (lab, col, members) in enumerate(FAMILIES):
        jit = 1 + 0.03 * (j - 2.5)
        pts = [(dofx(NP[n] - 2) * jit, joint_alpha(arm, n), n) for n in members if n in NP]
        pts = [q for q in pts if q[1] is not None]
        if pts:
            ax.plot([q[0] for q in pts], [q[1] for q in pts], ls=":", marker="o", color=col, zorder=3, label=lab)
        for n in members:
            a = solo_alpha(n)
            if a is not None:
                ax.scatter([dofx(NP[n] - 2) * jit], [a], s=40, marker="s", facecolors="none", edgecolors=col, zorder=4)
        print("  " + lab + ": " + ", ".join(f"{q[2]} {q[1]:.2f}" for q in pts)
              + "".join(f" | {n} alone {solo_alpha(n):.2f}" for n in members if solo_alpha(n) is not None))
    ax.xaxis.set_major_locator(FixedLocator([dofx(n) for n in (2, 3, 4)])); ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xticklabels(["2", "3", "4"]); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2])); ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("final-state particles"); ax.set_ylabel(r"$\alpha_C$")
    ps.process_label(ax, "arithmetic mean" if arm == "arith" else "geometric mean", loc="upper right")
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    # the bound labelled along its line, as in fig:alphamult (the sanctioned second in-axes label)
    x0 = np.sqrt(XLIM[0] * XLIM[1])
    p = lambda x: ax.transData.transform((x, 4.0 / x))
    (dx, dy) = p(x0 * 1.3) - p(x0 / 1.3)
    ax.text(x0, (4.0 / x0) * 0.86, "Theoretical lower bound", color="0.55",
            rotation=np.degrees(np.arctan2(dy, dx)), rotation_mode="anchor", ha="center", va="center", zorder=1)
# six family labels and the marker key cannot sit inside a 2.40in box (fig:alphamult hit the same
# with eight): one strip above the two panels, paid for by the canvas, as that figure does
h0, l0 = axes[0].get_legend_handles_labels()
k1 = axes[0].plot([], [], ls="none", marker="o", color="0.3")[0]
k2 = axes[0].scatter([], [], s=40, marker="s", facecolors="none", edgecolors="0.3")
ps.shared_legend(fig, axes[0], ncol=4, handles=h0 + [k1, k2], labels=l0 + ["joint run", "trained alone"])
ps.save(fig, os.path.join(HERE, "alpha_vs_multiplicity_steps"))
