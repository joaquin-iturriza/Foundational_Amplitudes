"""Transfer study, the large-D end: each probe's scratch curve up to D = 10^4 (best trial of the 8k-step
search, the t-channel factor off where it applies: tp2_scr, else tp_scr) with one 64k-step run at
D = 10^4.5 and 10^5 (fixed HPs, lr 1.2e-3, the factor off; scripts/job_transfer_calib.sh). The two
compute budgets differ, so the 64k points are drawn apart (squares). Runs that timed out are listed in
long_runs.json with the step they reached and are not drawn. Loss is MSE of log|M|^2 at the best
checkpoint.
  <base>_a  the t/u-channel 2->2 probes (uu~->gg, uu~->Zg, ee->Za)
  <base>_b  the probes with a Z pole (ee->nu_e nu_e~, and ee->dd~, ee->bb~ at one loop)
Data: analysis/transfer/long_runs.json, scratch_sweeps.json.
    python analysis/transfer/large_d.py
"""
import json, os, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps

L = json.load(open(os.path.join(ROOT, "analysis", "transfer", "long_runs.json")))
S = json.load(open(os.path.join(ROOT, "analysis", "transfer", "scratch_sweeps.json")))
LAB = {"uubar_gg": r"$u\bar u\to gg$", "uubar_Zg": r"$u\bar u\to Zg$", "ee_Za": r"$e^+e^-\to Z\gamma$",
       "ee_nnbar": r"$e^+e^-\to\nu_e\bar\nu_e$", "ee_dd_nlo": r"$e^+e^-\to d\bar d$ (1-loop)",
       "ee_bb_nlo": r"$e^+e^-\to b\bar b$ (1-loop)"}
GROUPS = [["uubar_gg", "uubar_Zg", "ee_Za"], ["ee_nnbar", "ee_dd_nlo", "ee_bb_nlo"]]


def curve(p):
    fam = "tp2_scr" if any(n.startswith(f"tp2_scr_{p}_d") for n in S) else "tp_scr"
    D, y = [], []
    for k in range(2, 9):
        tr = [t for n, v in S.items() if n == f"{fam}_{p}_d{k}" or n.startswith(f"{fam}_{p}_d{k}_") for t in v]
        if tr:
            b = min(tr, key=lambda t: t["val_loss"])
            D.append(10 ** (k / 2)); y.append(b["val_loss"] * b["prepd_std"] ** 2)
    return D, y


def long64(p):
    D, y = [], []
    for k in (9, 10):
        for n, r in L.items():
            if re.fullmatch(rf"tp2?_calib_{p}_d{k}_t64000_lr[0-9.e-]+", n) and not r.get("failed") \
                    and str(r.get("tchannel")).lower() == "false":
                D.append(10 ** (k / 2)); y.append(r["val"] * r["std"] ** 2)
    return D, y


figs = ps.panels(2)
for (fig, ax), group in zip(figs, GROUPS):
    for p, col in zip(group, ps.CYCLE):
        D, y = curve(p)
        ax.plot(D, y, "o-", color=col, label=LAB[p])
        D64, y64 = long64(p)
        ax.plot(D64, y64, "s", color=col, mfc="none", ls="none")
        print(p, " ".join(f"{d:.0f}:{v:.2g}" for d, v in zip(D + D64, y + y64)))
    ax.plot([], [], "s", color="k", mfc="none", ls="none", label="64k steps")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.legend(ax, "lower left")
    ps.make_room(ax)
ps.save_panels(figs, os.path.join(ROOT, "analysis", "transfer", "figs", "large_d"))
