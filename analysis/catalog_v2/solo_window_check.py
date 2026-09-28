"""The two solo scaling sets on one axis, for the processes they share (ee->aa, uu, uug, uugg): the old
from-scratch set of tab:scaling (sweeps/scaling_solo_full on Jean Zay: 70k train events, batch 16384,
4 heads, 4 ... 4000 steps, 15 DyHPO trials per cell; collected into solo_full_old.json by
fit_scaling_law.collect_best_from_config's cell layout) and the batch-1024 solo references (solo_b1k:
full catalog pools, batch 1024, 8 heads, 33 ... 1072 steps, 6 trials per cell). Each cell at its best
trial's val_loss (the DyHPO result; CLAUDE.md, Reported values).
    python analysis/catalog_v2/solo_window_check.py
Writes solo_window_check (a: against events seen, b: against optimizer steps) and prints the
floor-aware fit of each curve (CLAUDE.md, Scaling fits) and the local slopes between neighbouring cells."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "sweep")]
import plot_style as ps
from solo_b1k import solo_mse
from solo_datalimit_labels import LABEL
from analyze_pretraining_scaling import fit_power_law_with_floor
OLD = json.load(open(os.path.join(HERE, "solo_full_old.json")))
NEW = solo_mse()
PROCS = [("ee_aa", ps.C.blue), ("ee_uu", ps.C.sky), ("ee_uug", ps.C.orange), ("ee_uugg", ps.C.vermillion)]
SETS = [("old", OLD, 16384, [4, 126, 400, 1265, 4000], "o", "-", "batch 16384, 70k events (tab:scaling)"),
        ("new", NEW, 1024, [33, 67, 134, 268, 536, 1072], "s", "--", "batch 1024, catalog pool")]
val = lambda s, D, p, t: min(D[f"{p}|{t}"]) if s == "old" else D[f"{p}|{t}"]
fig, (axE, axS) = ps.figure(ncols=2)
for p, col in PROCS:
    for s, D, bs, T, mk, lsty, _ in SETS:
        y = np.array([val(s, D, p, t) for t in T]); t = np.array(T, float)
        for ax, x in ((axE, bs * t), (axS, t)):
            ax.plot(x, y, marker=mk, ls=lsty, color=col, mfc=col if s == "old" else "none")
        f = fit_power_law_with_floor(t, y)
        loc = -np.diff(np.log(y)) / np.diff(np.log(t))
        print(f"{p:8s} {s}: floor-aware alpha {f[1]:.2f} (L_inf {f[2]:.2g}); local slopes " + " ".join(f"{v:.1f}" for v in loc))
h = [axE.plot([], [], color=col, ls="-", label=LABEL.get(p, p))[0] for p, col in PROCS]
h += [axE.plot([], [], color="black", marker=mk, ls=lsty, mfc="black" if s == "old" else "none", label=lab)[0] for s, _, _, _, mk, lsty, lab in SETS]
for ax, xl in ((axE, "events seen"), (axS, "optimizer steps")):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xl); ax.set_ylabel(r"MSE($\log|\mathcal{M}|^2$)")
ps.legend(axE, "lower left", handles=h)
ps.save(fig, "analysis/catalog_v2/solo_window_check")
