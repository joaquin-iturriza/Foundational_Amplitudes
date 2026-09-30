"""Transfer study, horizon test: the best trial of a cell's 8k-step search (tp2_scr, tp_scr for
ee->tt~, where the t-channel factor never applies) re-run at 32k steps with the same HPs
(scripts/job_transfer_calib.sh), at D = 10^3.5 and 10^4. Loss is MSE of log|M|^2 at the best
checkpoint (val_loss_no_reg times the run's prepd_std^2). A run that diverged is drawn at its best
validation before the divergence (from its log, recorded in long_runs.json with the step), marked x:
its lr had not annealed, so that value is not a converged one.
  <base>_a, _b  loss against the horizon at D = 10^3.5 and 10^4
  <base>_curves the 32k runs' validation curves
Data: analysis/transfer/long_runs.json, scratch_sweeps.json.
    python analysis/transfer/horizon_test.py
"""
import json, os, re, sys
import numpy as np
from matplotlib.ticker import NullLocator
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "analysis", "transfer", "figs")
sys.path.insert(0, ROOT)
import plot_style as ps

L = json.load(open(os.path.join(ROOT, "analysis", "transfer", "long_runs.json")))
S = json.load(open(os.path.join(ROOT, "analysis", "transfer", "scratch_sweeps.json")))
PROBES = {"ud_ud": r"$ud\to ud$", "uubar_Zg": r"$u\bar u\to Zg$", "ee_ttbar": r"$e^+e^-\to t\bar t$",
          "ee_Za": r"$e^+e^-\to Z\gamma$"}


def best8k(p, k):
    fam = "tp_scr" if p == "ee_ttbar" else "tp2_scr"
    b = min(S[f"{fam}_{p}_d{k}"], key=lambda t: t["val_loss"])
    return b["val_loss"] * b["prepd_std"] ** 2


def run32k(p, k):
    for n, r in L.items():
        if re.fullmatch(rf"tp2_calib_{p}_d{k}_t32000_lr[0-9.e-]+", n):
            if r.get("failed"):
                m = re.search(r"before it ([0-9.e+-]+) \(std ([0-9.]+)\)", r["failed"])
                return float(m.group(1)) * float(m.group(2)) ** 2, True, n
            return r["val"] * r["std"] ** 2, False, n
    return None


figs = ps.panels(2)
for (fig, ax), k in zip(figs, (7, 8)):
    for (p, lab), col in zip(PROBES.items(), ps.CYCLE):
        r = run32k(p, k)
        if r is None:
            continue
        y8, (y32, div, n) = best8k(p, k), r
        ax.plot([8000, 32000], [y8, y32], "--" if div else "-", color=col, label=lab)
        ax.plot(8000, y8, "o", color=col, ls="none")
        ax.plot(32000, y32, "x" if div else "o", color=col, ls="none")
        print(f"{p} D=10^{k / 2:g}: 8k {y8:.3g}  32k {y32:.3g}{' (diverged, best before)' if div else ''}  {y8 / y32:.2f}x")
    ax.plot([], [], "x", color="k", ls="none", label="diverged (best before)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([8000, 32000], ["8k", "32k"]); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(6000, 42000)
    ax.set_xlabel("horizon (steps)"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(ax, rf"$D=10^{{{k / 2:g}}}$")
    ps.legend(ax, "lower left")
    ps.make_room(ax)
ps.save_panels(figs, os.path.join(FIG, "horizon_test"))

fig, ax = ps.figure()
rows = []
for (p, lab), col in zip(PROBES.items(), ps.CYCLE):
    for k, ls in ((7, "-"), (8, ":")):
        for n, r in L.items():
            if re.fullmatch(rf"tp2_calib_{p}_d{k}_t32000_lr[0-9.e-]+", n) and r.get("curve"):
                y = np.array(r["curve"]) * r["std"] ** 2
                ax.plot(r["every"] * np.arange(1, len(y) + 1), y, ls, color=col,
                        label=rf"{lab}, $10^{{{k / 2:g}}}$")
ax.set_yscale("log")
ax.set_xlabel("step"); ax.set_ylabel(r"val MSE$(\log|\mathcal{M}|^2)$")
ps.shared_legend(fig, ax, ncol=2)
ps.save(fig, os.path.join(FIG, "horizon_test_curves"))
