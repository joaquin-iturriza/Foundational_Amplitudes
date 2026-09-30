"""Breit-Wigner factor off (tp3_, data.target_propagators false) at each cell's best HPs, one run per cell
(scripts/job_transfer_calib.sh fam=tp3_scr), against the best of the cell's 8-trial factor-on search
(tp_scr; for ee->nu_e nu_e~ also tp2_scr, whose target is the same). Loss is MSE of log|M|^2 at the best
checkpoint; the factor is exact, so the two arms share the unit. Note the comparison is not paired
fairly: the factor-on value is the minimum over eight trials of a noisy search, the factor-off value a
single draw at the winner's HPs and seed.
Data: analysis/transfer/long_runs.json (tp3_calib_*), scratch_sweeps.json.
    python analysis/transfer/bw_off_shortcut.py
"""
import json, os, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps

L = json.load(open(os.path.join(ROOT, "analysis", "transfer", "long_runs.json")))
S = json.load(open(os.path.join(ROOT, "analysis", "transfer", "scratch_sweeps.json")))
FAMS = {"ee_ddbar": ["tp_scr"], "ee_nnbar": ["tp_scr", "tp2_scr"], "ee_dd_nlo": ["tp_scr"], "ee_bb_nlo": ["tp_scr"]}
LAB = {"ee_ddbar": r"$e^+e^-\to d\bar d$", "ee_nnbar": r"$e^+e^-\to\nu_e\bar\nu_e$",
       "ee_dd_nlo": r"$e^+e^-\to d\bar d$ (1-loop)", "ee_bb_nlo": r"$e^+e^-\to b\bar b$ (1-loop)"}
figs = ps.panels(4)
for (fig, ax), p in zip(figs, FAMS):
    D, on, off = [], [], []
    for k in range(2, 9):
        tr = [t for f in FAMS[p] for n, v in S.items() if n == f"{f}_{p}_d{k}" or n.startswith(f"{f}_{p}_d{k}_") for t in v]
        run = [r for n, r in L.items() if re.fullmatch(rf"tp3_calib_{p}_d{k}_t\d+_lr[0-9.e-]+", n) and not r.get("failed")]
        if not tr or not run:
            continue
        b = min(tr, key=lambda t: t["val_loss"])
        D.append(10 ** (k / 2)); on.append(b["val_loss"] * b["prepd_std"] ** 2); off.append(run[0]["val"] * run[0]["std"] ** 2)
    ax.plot(D, on, "o-", color=ps.C.blue, label="factor on, best of 8")
    ax.plot(D, off, "o-", color=ps.C.vermillion, label="factor off, 1 run at those HPs")
    print(p, " ".join(f"{a / b:.2f}" for a, b in zip(on, off)))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(ax, LAB[p])
    ps.legend(ax, "lower left")
    ps.make_room(ax)
ps.save_panels(figs, os.path.join(ROOT, "analysis", "transfer", "figs", "bw_off_shortcut"))
