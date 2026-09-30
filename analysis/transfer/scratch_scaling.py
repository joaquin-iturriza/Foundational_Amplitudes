"""Transfer pilot, scratch curves: each probe alone on D = 10^(k/2) events, one single-fidelity
DyHPO per cell (sweeps tp_scr_<probe>_d<k>, horizons from the calibration). Loss is MSE of
log|M|^2 at the best checkpoint: val_loss_no_reg times the run's prepd_std^2 (each run standardizes
on its own D events).
  <base>_a / _b   ee->dd~ / uu~->gg vs D: every trial, each cell's best trial, and for ee->dd~ the
                  calibration run (one fixed HP point, analysis/transfer/calib_ee_ddbar.json);
                  dashed, the floor-aware law L = A D^-alpha + L_inf over the best trials
  <curves>_a / _b the best trial's validation curve per D, against steps
Data: analysis/transfer/scratch_sweeps.json (collect_sweeps.py tp_scr_ on the site).
    python analysis/transfer/scratch_scaling.py
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "sweep")]
import plot_style as ps
from analyze_pretraining_scaling import fit_power_law_with_floor

S = json.load(open(os.path.join(ROOT, "analysis", "transfer", "scratch_sweeps.json")))
CAL = json.load(open(os.path.join(ROOT, "analysis", "transfer", "calib_ee_ddbar.json")))
PROBES = [("ee_ddbar", r"$e^+e^-\to d\bar d$"), ("uubar_gg", r"$u\bar u\to gg$")]
FIG = os.path.join(ROOT, "analysis", "transfer", "figs")
phys = lambda t: t["val_loss"] * t["prepd_std"] ** 2


def cells(p):
    out = {}
    for name, trials in S.items():
        if name.startswith(f"tp_scr_{p}_d") and trials:
            k = int(name.split("_d")[-1].split("_")[0])
            out.setdefault(k, []).extend(trials)       # a resubmitted cell (_002) adds its trials
    return dict(sorted(out.items()))


figs = ps.panels(2)
for (fig, ax), (p, lab) in zip(figs, PROBES):
    C = cells(p)
    D = {k: round(10 ** (k / 2)) for k in C}
    xs = [D[k] for k in C for _ in C[k]]; ys = [phys(t) for k in C for t in C[k]]
    ax.plot(xs, ys, ".", color=ps.C.grey, ls="none", label="trials")
    best = {k: min(C[k], key=lambda t: t["val_loss"]) for k in C}
    bx = np.array([D[k] for k in best]); by = np.array([phys(best[k]) for k in best])
    ax.plot(bx, by, "o", color=ps.C.blue, ls="none", label="HPO best")
    if p == "ee_ddbar":
        cx = np.array([round(10 ** (r["k"] / 2)) for r in CAL if r["k"] in C])
        cy = np.array([r["val"] * r["prepd_std"][0] ** 2 for r in CAL if r["k"] in C])
        ax.plot(cx, cy, "s", mfc="none", color=ps.C.vermillion, ls="none", label="fixed HP (calibration)")
    f = fit_power_law_with_floor(bx, by)
    if f:
        x = np.geomspace(bx.min(), bx.max(), 200)
        ax.plot(x, f[0] * x ** -f[1] + f[2], color="k", ls="--",
                label=rf"$A D^{{-\alpha}}+L_\infty$, $\alpha={f[1]:.2f}$")
        print(f"{p}: alpha={f[1]:.3f} A={f[0]:.3g} L_inf={f[2]:.3g} chi2r={f[3]:.3g}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(ax, lab)
    ps.legend(ax, "lower left")
    for k in C:
        b = best[k]
        print(f"  {p} D={D[k]:>6}  n={len(C[k])}  best={phys(b):.3g}  best_step={b.get('best_step')}/{b['T']}  lr={b['lr']:.3g}")
ps.save_panels(figs, os.path.join(FIG, "scratch_scaling"))

figs = ps.panels(2)
for (fig, ax), (p, lab) in zip(figs, PROBES):
    C = cells(p)
    cols = ps.sequence(len(C))
    for col, (k, trials) in zip(cols, C.items()):
        b = min(trials, key=lambda t: t["val_loss"])
        if not b.get("val_curve"):
            continue
        y = np.array(b["val_curve"]) * b["prepd_std"] ** 2
        x = b["validate_every"] * np.arange(1, len(y) + 1)
        ax.plot(x, y, color=col, label=rf"$10^{{{k / 2:g}}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel(r"val MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(ax, lab)
    ps.legend(ax, "lower left", ncol=2, title=r"$D$")
ps.save_panels(figs, os.path.join(FIG, "scratch_curves"))
