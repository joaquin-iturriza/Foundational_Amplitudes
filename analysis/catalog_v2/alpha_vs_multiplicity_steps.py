"""Scaling exponent against final-state multiplicity for the catalog steps curve, in the style of
analysis/scaling_compute/alpha_vs_multiplicity_overlay.py (fig:alphamult): log alpha against the
phase-space DOF = 3 n_fs - 4 on a log axis (ticks by n_fs), the bound alpha = 4/DOF as a shaded wedge.
Joint: every process of the catalog fitted on its own points (1000-8000 steps x three seeds, full
pools; analysis/catalog_v2/steps_tuned.json) with the floor-aware law A C^-alpha + L_inf, then the
median and interquartile range per n_fs, trees and one-loop apart, for both training aggregations.
Alone: the reference processes at bs 1024 on the full pools (sweeps/solob1k_t*), one fit each; the
ee -> u ubar (+g, +gg) chain joined as a family. Processes whose fit fails are counted and left out.
    python analysis/catalog_v2/alpha_vs_multiplicity_steps.py
Writes analysis/catalog_v2/alpha_vs_multiplicity_steps (png + pdf)."""
import json, os, sys
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "sweep"))
import plot_style as ps
from analyze_pretraining_scaling import fit_power_law_with_floor, flops_per_step
from solo_datalimit_labels import LABEL

D = json.load(open(os.path.join(HERE, "steps_tuned.json")))
cls = json.load(open(os.path.join(HERE, "steps_agg.json")))["cls"].get
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
FIT_STEPS, SOLO_STEPS = [1000, 2000, 4000, 8000], [33, 67, 134, 268, 536, 1072]
runs = [r for r in D["joint"] if np.median(list(r["final"].values())) <= 0.5 and r["steps"] in FIT_STEPS]
names = sorted({n for r in runs for n in r["final"]} & set(NP))
nbar = float(np.mean([NP[n] for n in names]))
loop = lambda n: (cls(n) or "").endswith("1-loop")

def alpha(c, l):
    f = fit_power_law_with_floor(c, l)
    return None if f is None else f[1]

# joint: one fit per process and aggregation
A = {}
for arm in ("arith", "geo"):
    rs = [r for r in runs if r["arm"] == arm]
    for n in names:
        pts = [(flops_per_step(8, nbar, 16384) * r["steps"], r["final"][n]) for r in rs if n in r["final"]]
        a = alpha([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 4 else None
        A[arm, n] = a
    bad = sum(A[arm, n] is None for n in names)
    print(f"{arm}: {len(names) - bad}/{len(names)} per-process fits")
# alone
S = {}
for p in LABEL:
    t = [s for s in SOLO_STEPS if f"{p}|{s}" in D["solo"]]
    S[p] = alpha([flops_per_step(8, NP[p], 1024) * s for s in t], [D["solo"][f"{p}|{s}"] for s in t])

dofx = lambda n: 3.0 * n - 4.0
fig, ax = ps.figure()
ax.set_xscale("log"); ax.set_yscale("log")
XLIM = (1.55, 10.5); YLIM = (0.1, 4.0 / XLIM[0])
dd = np.array(XLIM)
ax.plot(dd, 4.0 / dd, color="0.65", ls="--", zorder=1, label=r"bound $\alpha=4/\mathrm{DOF}$")
ax.fill_between(dd, YLIM[0], 4.0 / dd, color="0.5", alpha=0.14, zorder=0, lw=0)

print(f"{'':12s} " + " ".join(f"{'n_fs=' + str(k):>22s}" for k in (2, 3, 4)))
SER = [("arith", False, ps.C.blue, "o", "joint, arithmetic, tree"), ("arith", True, ps.C.blue, "^", "joint, arithmetic, one-loop"),
       ("geo", False, ps.C.vermillion, "o", "joint, geometric, tree"), ("geo", True, ps.C.vermillion, "^", "joint, geometric, one-loop")]
for i, (arm, lp, col, mk, lab) in enumerate(SER):
    xs, med, lo, hi, cells = [], [], [], [], []
    for k in (2, 3, 4):
        v = [A[arm, n] for n in names if NP[n] - 2 == k and loop(n) == lp and A[arm, n] is not None]
        if len(v) < 3: cells.append(""); continue
        q1, m, q3 = np.percentile(v, [25, 50, 75])
        xs.append(dofx(k) * (1 + 0.035 * (i - 1.5))); med.append(m); lo.append(m - q1); hi.append(q3 - m)
        cells.append(f"{m:.2f} [{q1:.2f},{q3:.2f}] ({len(v)})")
    print(f"{lab:28s} " + " ".join(f"{c:>22s}" for c in cells))
    ax.errorbar(xs, med, yerr=[lo, hi], fmt=mk, color=col, capsize=2, ls=":", label=lab)
chain = ["ee_uu", "ee_uug", "ee_uugg"]
ax.plot([dofx(NP[p] - 2) for p in chain if S[p]], [S[p] for p in chain if S[p]], ls=":", color=ps.C.grey, zorder=2)
for p, a in S.items():
    if a is None: continue
    ax.scatter([dofx(NP[p] - 2) * 1.10], [a], s=30, marker="^" if p.endswith("_nlo") else "s",
               color=ps.C.grey, zorder=3)
ax.scatter([], [], s=30, marker="s", color=ps.C.grey, label="alone, tree (reference processes)")
ax.scatter([], [], s=30, marker="^", color=ps.C.grey, label="alone, one-loop")
print("alone: " + ", ".join(f"{p} (n_fs {NP[p]-2}) {a:.2f}" for p, a in S.items() if a))
ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
ax.xaxis.set_major_locator(FixedLocator([dofx(n) for n in (2, 3, 4)])); ax.xaxis.set_minor_locator(FixedLocator([]))
ax.set_xticklabels(["2", "3", "4"]); ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2])); ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("final-state particles"); ax.set_ylabel(r"$\alpha_C$ in $A\,C^{-\alpha_C}+L_\infty$")
ps.shared_legend(fig, ax, ncol=1)
ps.save(fig, os.path.join(HERE, "alpha_vs_multiplicity_steps"))
