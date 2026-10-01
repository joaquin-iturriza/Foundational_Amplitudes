"""Transfer study, seed noise of the scratch curves: each cell's best trial (seed 42, the minimum of an
8-trial search) re-run at the same HPs with seeds 1 and 2 (scripts/job_transfer_calib.sh seed=), for the
seven probes whose target no open question touches. Line: geometric mean of the two re-run seeds (an
unbiased draw at those HPs); band: their range together with the search best; marker: the search best,
which is a minimum over eight trials and so sits low by selection. Loss is MSE of log|M|^2 at the best
checkpoint.
  <base>_a  uu~->gg, uu~->Zg, ee->Za, ud->ud
  <base>_b  uu~->Zgg, uu~->Zggg, ee->tt~
Data: analysis/transfer/long_runs.json (tp2_calib_*_seed1/2), scratch_sweeps.json.
    python analysis/transfer/seeds_scratch.py
"""
import json, os, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps

L = json.load(open(os.path.join(ROOT, "analysis", "transfer", "long_runs.json")))
S = json.load(open(os.path.join(ROOT, "analysis", "transfer", "scratch_sweeps.json")))
FAM = {"uubar_gg": "tp2_scr", "uubar_Zg": "tp2_scr", "ee_Za": "tp2_scr", "ud_ud": "tp2_scr",
       "uubar_Zgg": "tp_scr", "uubar_Zggg": "tp_scr", "ee_ttbar": "tp_scr"}
LAB = {"uubar_gg": r"$u\bar u\to gg$", "uubar_Zg": r"$u\bar u\to Zg$", "ee_Za": r"$e^+e^-\to Z\gamma$",
       "ud_ud": r"$ud\to ud$", "uubar_Zgg": r"$u\bar u\to Zgg$", "uubar_Zggg": r"$u\bar u\to Zggg$",
       "ee_ttbar": r"$e^+e^-\to t\bar t$"}
GROUPS = [["uubar_gg", "uubar_Zg", "ee_Za", "ud_ud"], ["uubar_Zgg", "uubar_Zggg", "ee_ttbar"]]
figs = ps.panels(2)
for (fig, ax), group in zip(figs, GROUPS):
    for p, col in zip(group, ps.CYCLE):
        D, best, gm, lo, hi = [], [], [], [], []
        for k in range(2, 9):
            tr = [t for n, v in S.items() if n == f"{FAM[p]}_{p}_d{k}" or n.startswith(f"{FAM[p]}_{p}_d{k}_") for t in v]
            sd = [r["val"] * r["std"] ** 2 for n, r in L.items()
                  if re.fullmatch(rf"tp2_calib_{p}_d{k}_t\d+_lr[0-9.e-]+_seed[12]", n) and not r.get("failed")]
            if not tr or len(sd) < 2:
                continue
            b = min(t["val_loss"] * t["prepd_std"] ** 2 for t in tr)
            D.append(10 ** (k / 2)); best.append(b); gm.append(float(np.exp(np.mean(np.log(sd)))))
            lo.append(min(sd + [b])); hi.append(max(sd + [b]))
        ax.fill_between(D, lo, hi, color=col, alpha=0.2, lw=0)
        ax.plot(D, gm, "-", color=col, label=LAB[p])
        ax.plot(D, best, "o", color=col, mfc="none", ls="none")
        print(p, " ".join(f"{d:.0f}:{g:.2g}[{l:.2g},{h:.2g}]" for d, g, l, h in zip(D, gm, lo, hi)))
    ax.plot([], [], "o", color="k", mfc="none", ls="none", label="search best (seed 42)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.legend(ax, "lower left")
    ps.make_room(ax)
ps.save_panels(figs, os.path.join(ROOT, "analysis", "transfer", "figs", "seeds_scratch"))
