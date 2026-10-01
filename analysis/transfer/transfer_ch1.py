"""Transfer study, chapter one: pretrain on ee->uu alone (tp3_pre_ee_uu, both target factors off, the
fixed-HP run at the search's best), fine-tune each probe on D = 10^(k/2) events (tp3_ft), against the
same probe from scratch. Best trial of each cell's single-fidelity DyHPO, MSE of log|M|^2 at the best
checkpoint (val_loss_no_reg times the run's prepd_std^2). Scratch and fine-tune of a probe share the
target: ee->dd~ the tp3_scr sweeps (factors off in both), uu~->gg the tp2_scr sweeps (no Breit-Wigner
factor reaches its pool, so the tp2_ and tp3_ targets are the same). Only the D values where both arms have a cell are drawn (a fine-tune
is compared with its scratch cell, nothing else). Open markers: cells with fewer than 7 trials in.
  <base>_a  ee->dd~ (near probe)
  <base>_b  uu~->gg (far probe)
Data: analysis/transfer/scratch_sweeps.json (collect_sweeps.py tp on each site, merged).
    python analysis/transfer/transfer_ch1.py
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "analysis", "transfer", "figs")
sys.path.insert(0, ROOT)
import plot_style as ps

S = json.load(open(os.path.join(ROOT, "analysis", "transfer", "scratch_sweeps.json")))


def cells(fam, p):
    out = {}
    for name, trials in S.items():
        if name.startswith(f"{fam}_{p}_d") and trials:
            out.setdefault(int(name[len(f"{fam}_{p}_d"):].split("_")[0]), []).extend(trials)   # _002 resubmissions merge
    return dict(sorted(out.items()))


figs = ps.panels(2)
for (fig, ax), (p, scr, lab) in zip(figs, (("ee_ddbar", "tp3_scr", r"$e^+e^-\to d\bar d$"),
                                            ("uubar_gg", "tp2_scr", r"$u\bar u\to gg$"))):
    both = set(cells(scr, p)) & set(cells("tp3_ft", p))
    for fam, col, name in ((scr, ps.C.blue, "from scratch"), ("tp3_ft", ps.C.vermillion, "fine-tuned")):
        C = {k: t for k, t in cells(fam, p).items() if k in both}
        D = np.array([10 ** (k / 2) for k in C])
        L = np.array([min(r["val_loss"] * r["prepd_std"] ** 2 for r in t) for t in C.values()])
        done = np.array([len(t) >= 7 for t in C.values()])
        ax.plot(D, L, "-", color=col, label=name)
        ax.plot(D[done], L[done], "o", color=col, ls="none")
        ax.plot(D[~done], L[~done], "o", color=col, mfc="none", ls="none")
        print(p, fam, " ".join(f"{d:.0f}:{l:.2g}({len(t)})" for d, l, t in zip(D, L, C.values())))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(ax, lab)
    ps.legend(ax, "lower left")
ps.save_panels(figs, os.path.join(FIG, "transfer_ch1"))
