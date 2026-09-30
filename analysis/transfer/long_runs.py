"""Transfer pilot, longer horizons: fixed-HP scratch runs (scripts/job_transfer_calib.sh) of one
probe at one D, at 8k/16k/32k steps, against the cell's 8k HPO best (tp_scr sweeps). Loss is MSE of
log|M|^2 at the best checkpoint (val_loss_no_reg times the run's prepd_std^2).
  <base>_a  ee->dd~ : loss vs horizon per D; marker fill = the run's lr (the 32k runs took a
            1.8x higher lr than the 16k runs, so that step is not a clean horizon comparison)
  <base>_b  uu~->gg : loss vs horizon per D (fixed HPs at 16k/32k = the cell's 8k HPO best for
            D = 10^3.5, 10^4; the window centre for 10^4.5, 10^5); open = the 8k HPO best
  <curves>  uu~->gg : the long runs' validation curves
Only runs with the t-channel factor on (as the tp_scr sweeps they are compared with).
Data: analysis/transfer/long_runs.json (analysis/transfer/collect_calib.py on each site, merged),
analysis/transfer/scratch_sweeps.json.
    python analysis/transfer/long_runs.py
"""
import json, os, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps

L = json.load(open(os.path.join(ROOT, "analysis", "transfer", "long_runs.json")))
S = json.load(open(os.path.join(ROOT, "analysis", "transfer", "scratch_sweeps.json")))
FIG = os.path.join(ROOT, "analysis", "transfer", "figs")
phys = lambda r: r["val"] * r["std"] ** 2
runs = []
for n, r in L.items():
    m = re.match(r"tp2?_calib_(\w+?)_d(\d+)_t(\d+)", n)
    # this figure is the factor-on pilot (the horizon question): the arm is read from what ran
    # (collect_calib.py records the run's own target_propagator_tchannel), never from the name
    if m and r.get("std") and str(r.get("tchannel", True)).lower() == "true" \
            and not re.search(r"target_propagators-false", n):
        runs.append(dict(p=m.group(1), k=int(m.group(2)), T=int(m.group(3)), **r))


def hpo_best(p, k):
    tr = [t for name, v in S.items() if re.fullmatch(rf"tp_scr_{p}_d{k}(_\d+)?", name) for t in v]
    if not tr:
        return None
    b = min(tr, key=lambda t: t["val_loss"])
    return b["T"], b["val_loss"] * b["prepd_std"] ** 2


figs = ps.panels(2)
for (fig, ax), p, lab, ks in ((figs[0], "ee_ddbar", r"$e^+e^-\to d\bar d$", (8, 9, 10)),
                              (figs[1], "uubar_gg", r"$u\bar u\to gg$", (7, 8, 9, 10))):
    for k, col in zip(ks, ps.CYCLE):
        rs = sorted([r for r in runs if r["p"] == p and r["k"] == k], key=lambda r: r["T"])
        if not rs:
            continue
        ax.plot([r["T"] for r in rs], [phys(r) for r in rs], "-", color=col, label=rf"$D=10^{{{k / 2:g}}}$")
        for r in rs:
            hi = p == "ee_ddbar" and r["lr"] > 2.5e-3
            ax.plot(r["T"], phys(r), "s" if hi else "o", color=col, ls="none")
        hb = hpo_best(p, k)
        if hb:
            ax.plot(*hb, "o", mfc="none", color=col, ls="none")
    if p == "ee_ddbar":
        ax.plot([], [], "o", color="k", ls="none", label=r"lr $\approx1.6\times10^{-3}$")
        ax.plot([], [], "s", color="k", ls="none", label=r"lr $\approx3.0\times10^{-3}$")
    ax.plot([], [], "o", mfc="none", color="k", ls="none", label="8k HPO best")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("horizon (steps)"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(ax, lab)
    ps.legend(ax, "lower left")
ps.save_panels(figs, os.path.join(FIG, "long_runs"))

fig, ax = ps.figure()
rs = sorted([r for r in runs if r["p"] == "uubar_gg" and r.get("curve")], key=lambda r: (r["k"], r["T"]))
cols = ps.sequence(len(rs))
for col, r in zip(cols, rs):
    y = np.array(r["curve"]) * r["std"] ** 2
    ax.plot(r["every"] * np.arange(1, len(y) + 1), y, color=col, label=rf"$10^{{{r['k'] / 2:g}}}$, {r['T'] // 1000}k")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("step"); ax.set_ylabel(r"val MSE$(\log|\mathcal{M}|^2)$")
ps.process_label(ax, r"$u\bar u\to gg$")
ps.shared_legend(fig, ax, ncol=3)
ps.save(fig, os.path.join(FIG, "long_runs_curves_uubar_gg"))
