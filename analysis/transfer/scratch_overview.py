"""Transfer study, scratch curves of every probe so far: each probe alone on D = 10^(k/2) events,
the best trial of the cell's single-fidelity DyHPO, MSE of log|M|^2 at the best checkpoint
(val_loss_no_reg times the run's prepd_std^2). Open markers: cells whose sweep has not finished
(fewer than 7 trials in; a sweep has 8, and a duplicate suggestion can leave 7).
  <base>_a  the chapter-1 probes and the Z+ng family
  <base>_b  the ladder probes
Data: analysis/transfer/scratch_sweeps.json (collect_sweeps.py tp_ on each site, merged).
    python analysis/transfer/scratch_overview.py
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "analysis", "transfer", "figs")
sys.path.insert(0, ROOT)
import plot_style as ps
S = json.load(open(os.path.join(ROOT, "analysis", "transfer", "scratch_sweeps.json")))
LABEL = {"ee_ddbar": r"$e^+e^-\to d\bar d$", "uubar_gg": r"$u\bar u\to gg$",
         "uubar_Zg": r"$u\bar u\to Zg$", "uubar_Zgg": r"$u\bar u\to Zgg$", "uubar_Zggg": r"$u\bar u\to Zggg$",
         "ee_nnbar": r"$e^+e^-\to\nu_e\bar\nu_e$", "ee_Za": r"$e^+e^-\to Z\gamma$", "ud_ud": r"$ud\to ud$",
         "ee_ttbar": r"$e^+e^-\to t\bar t$", "ee_WW": r"$e^+e^-\to W^+W^-$",
         "ee_dd_nlo": r"$e^+e^-\to d\bar d$ (1-loop)", "ee_bb_nlo": r"$e^+e^-\to b\bar b$ (1-loop)"}
GROUPS = [["ee_ddbar", "uubar_gg", "uubar_Zg", "uubar_Zgg", "uubar_Zggg"],
          ["ee_nnbar", "ee_Za", "ud_ud", "ee_ttbar", "ee_WW", "ee_dd_nlo", "ee_bb_nlo"]]
def cells(p):
    out = {}
    for name, trials in S.items():
        if name.startswith(f"tp_scr_{p}_d") and trials:
            out.setdefault(int(name[len(f"tp_scr_{p}_d"):].split("_")[0]), []).extend(trials)   # _002 resubmissions merge
    return dict(sorted(out.items()))
figs = ps.panels(2)
for (fig, ax), group in zip(figs, GROUPS):
    for p, col in zip(group, ps.CYCLE):
        C = cells(p)
        if not C: continue
        D = np.array([10 ** (k / 2) for k in C])
        best = [min(t, key=lambda r: r["val_loss"]) for t in C.values()]
        L = np.array([b["val_loss"] * b["prepd_std"] ** 2 for b in best])
        done = np.array([len(t) >= 7 for t in C.values()])
        ax.plot(D, L, "-", color=col, label=LABEL[p])
        ax.plot(D[done], L[done], "o", color=col, ls="none")
        ax.plot(D[~done], L[~done], "o", color=col, mfc="none", ls="none")
        print(p, " ".join(f"{d:.0f}:{l:.2g}{'' if ok else '*'}({len(t)})" for d, l, ok, t in zip(D, L, done, C.values())))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.shared_legend(fig, ax, ncol=2)
ps.save_panels(figs, os.path.join(OUT, "scratch_overview"))
