"""Transfer pilot, horizon calibration: ee->dd~ from scratch at one fixed HP point per D
(scripts/job_transfer_calib.sh), D = 10^(k/2), k = 2..10. Loss is MSE of log|M|^2 at the best
checkpoint: the run's val_loss_no_reg (standardized units) times its prepd_std^2, since each run
standardizes on its own D events. Marker: the run's horizon (8000 or 16000 steps); open
markers are runs whose best checkpoint is in the last 2% of the horizon. Line: the floor-aware
law L = A D^-alpha + L_inf (docs/results.tex eq:scaling) over all nine points.
Data: analysis/transfer/calib_ee_ddbar.json (fetched from the runs' result JSON, data_stats.json
and best checkpoint).
    python analysis/transfer/calib_scaling.py
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "sweep")]
import plot_style as ps
from analyze_pretraining_scaling import fit_power_law_with_floor

rows = json.load(open(os.path.join(ROOT, "analysis", "transfer", "calib_ee_ddbar.json")))
D = np.array([round(10 ** (r["k"] / 2)) for r in rows], float)
L = np.array([r["val"] * r["prepd_std"][0] ** 2 for r in rows])
T = np.array([r["T"] for r in rows]); late = np.array([r["best_step"] >= 0.98 * r["T"] for r in rows])

fig, ax = ps.figure()
for t, col, mk in ((8000, ps.C.blue, "o"), (16000, ps.C.vermillion, "s")):
    for lt in (False, True):
        s = (T == t) & (late == lt)
        if s.any():
            ax.plot(D[s], L[s], mk, ls="none", color=col, mfc="none" if lt else col,
                    label=f"{t} steps" + (", best at the end" if lt else ""))
f = fit_power_law_with_floor(D, L)
if f:
    A, a, Linf = f[0], f[1], f[2]
    x = np.geomspace(D.min(), D.max(), 200)
    ax.plot(x, A * x ** -a + Linf, color="k", ls="--",
            label=rf"$A D^{{-\alpha}}+L_\infty$, $\alpha={a:.2f}$")
    print(f"floor-aware fit: A={A:.3g} alpha={a:.3f} L_inf={Linf:.3g} chi2r={f[3]:.3g}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"training events $D$")
ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
ps.process_label(ax, r"$e^+e^-\to d\bar d$")
ps.legend(ax, "lower left")
ps.save(fig, os.path.join(ROOT, "analysis", "transfer", "figs", "calib_scaling_ee_ddbar"))
for r, d, l in zip(rows, D, L):
    print(f"D={d:>7.0f}  steps={r['T']:>5}  best_step={r['best_step']:>5}  MSE(log|M|^2)={l:.3g}  GPU-h={r['hours']:.2f}")
