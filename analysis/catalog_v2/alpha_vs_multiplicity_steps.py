"""Scaling exponent against final-state multiplicity for the catalog steps curve: the same figure as
analysis/scaling_compute/alpha_vs_multiplicity_overlay.py (fig:alphamult) -- log alpha against the
phase-space DOF = 3 n_fs - 4 on a log axis (ticks by n_fs), the bound alpha = 4/DOF corner to corner
with a shaded wedge below and its label along the line -- for the twelve reference processes, the
ones trained both jointly and alone, coloured by kind (tree, resonant, positive and signed one-loop).
Dots: the process in the joint run (full pools, 1000-8000 steps x three seeds; steps_tuned.json),
fitted on its own points with the floor-aware law A C^-alpha + L_inf. Open squares: the same process
trained alone (bs 1024, full pool, sweeps/solob1k_t*); a thin line joins the two. One panel per training
aggregation: (a) arithmetic mean, (b) geometric mean.
    python analysis/catalog_v2/alpha_vs_multiplicity_steps.py
Writes analysis/catalog_v2/alpha_vs_multiplicity_steps (png + pdf)."""
import json, os, sys
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
import matplotlib.transforms as mtransforms
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "sweep"))
import plot_style as ps
from analyze_pretraining_scaling import fit_power_law_with_floor, flops_per_step

D = json.load(open(os.path.join(HERE, "steps_tuned.json")))
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
FIT_STEPS, SOLO_STEPS = [1000, 2000, 4000, 8000], [33, 67, 134, 268, 536, 1072]
runs = [r for r in D["joint"] if np.median(list(r["final"].values())) <= 0.5 and r["steps"] in FIT_STEPS]
nbar = float(np.mean([NP[n] for n in {n for r in runs for n in r["final"]} if n in NP]))
# the reference processes, by kind (the colours of the class panels)
KIND = {"tree": ("tree", ps.C.blue), "res": ("resonant", ps.C.sky), "pos": ("positive one-loop", ps.C.orange),
        "sgn": ("signed one-loop", ps.C.purple)}
PROCS = [("ee_aa", "tree"), ("uubar_uubar", "tree"), ("ee_uu", "res"), ("ee_ddbar", "res"),
         ("ee_uug", "tree"), ("udbar_WpZZ", "tree"), ("ee_uugg", "tree"), ("udbar_WpZaa", "tree"),
         ("uubar_ZaZ_nlo", "pos"), ("ee_bb_nlo", "pos"), ("udbar_Wgg_nlo", "sgn"), ("uubar_ddbara_nlo", "sgn")]

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
    # spread the processes that share a multiplicity so each joint/alone pair stands apart
    slot = {}
    for n, k in PROCS:
        m = NP[n] - 2; slot.setdefault(m, []).append(n)
    X = {n: dofx(m) * (1 + 0.07 * (i - (len(v) - 1) / 2)) for m, v in slot.items() for i, n in enumerate(v)}
    J = {n: joint_alpha(arm, n) for n, _ in PROCS}; S = {n: solo_alpha(n) for n, _ in PROCS}
    for n, k in PROCS:
        col = KIND[k][1]
        if J[n] is not None and S[n] is not None:
            ax.plot([X[n]] * 2, [J[n], S[n]], color=col, alpha=0.5, zorder=2)
        if J[n] is not None:
            ax.plot([X[n]], [J[n]], ls="none", marker="o", color=col, zorder=3)
        if S[n] is not None:
            ax.scatter([X[n]], [S[n]], s=40, marker="s", facecolors="none", edgecolors=col, zorder=4)
        print(f"  {n:18s} n_fs={NP[n]-2}  joint {J[n] if J[n] is None else round(J[n], 2)}  alone {S[n] if S[n] is None else round(S[n], 2)}")
    ax.xaxis.set_major_locator(FixedLocator([dofx(n) for n in (2, 3, 4)])); ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xticklabels(["2", "3", "4"]); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2])); ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("final-state particles"); ax.set_ylabel(r"$\alpha_C$")
    ps.process_label(ax, "arithmetic mean" if arm == "arith" else "geometric mean", loc="upper right")
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    # the bound labelled along its line, as in fig:alphamult (the sanctioned second in-axes label).
    # It runs corner to corner, so its on-page angle is fixed by the plot box (PLOT_W_IN x PLOT_H_IN,
    # which ps.save enforces); placed in axes coordinates, the label stays on the line after layout.
    ang = np.degrees(np.arctan2(-ps.PLOT_H_IN, ps.PLOT_W_IN))
    nx, ny = -ps.PLOT_H_IN, -ps.PLOT_W_IN                 # the normal pointing below the line
    k = 4.0 / np.hypot(nx, ny)                             # 4 pt clear of it
    ax.text(0.5, 0.5, "Theoretical lower bound", color="0.55", rotation=ang, rotation_mode="anchor",
            transform=mtransforms.offset_copy(ax.transAxes, fig=fig, x=nx * k, y=ny * k, units="points"),
            ha="center", va="top", zorder=1)
# four kinds and the marker key do not fit inside a 2.40in box: one strip above
# the two panels, as fig:alphamult does
H = [axes[0].plot([], [], ls="none", marker="o", color=c)[0] for _, c in KIND.values()]
L = [lab for lab, _ in KIND.values()]
H += [axes[0].plot([], [], ls="none", marker="o", color="0.3")[0],
      axes[0].scatter([], [], s=40, marker="s", facecolors="none", edgecolors="0.3")]
L += ["joint run", "trained alone"]
ps.shared_legend(fig, axes[0], ncol=4, handles=H, labels=L)
ps.save(fig, os.path.join(HERE, "alpha_vs_multiplicity_steps"))
