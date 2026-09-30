"""Transfer study, chapter one: pretrain on ee->uu alone (tp_pre_ee_uu, best trial), fine-tune each
probe on D = 10^(k/2) events, against the same probe from scratch. Best trial of each cell's
single-fidelity DyHPO, MSE of log|M|^2 at the best checkpoint (val_loss_no_reg times the run's
prepd_std^2). Scratch and fine-tune of a probe share the target: ee->dd~ the tp_ sweeps (the
Breit-Wigner factor on in both, the t-channel factor never applies), uu~->gg the tp2_ sweeps (the
t-channel factor off in both). Only the D values where both arms have a cell are drawn (a fine-tune
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
for (fig, ax), (p, gen, lab) in zip(figs, (("ee_ddbar", "tp", r"$e^+e^-\to d\bar d$"),
                                            ("uubar_gg", "tp2", r"$u\bar u\to gg$"))):
    both = set(cells(f"{gen}_scr", p)) & set(cells(f"{gen}_ft", p))
    for arm, col, name in (("scr", ps.C.blue, "from scratch"), ("ft", ps.C.vermillion, "fine-tuned")):
        C = {k: t for k, t in cells(f"{gen}_{arm}", p).items() if k in both}
        D = np.array([10 ** (k / 2) for k in C])
        L = np.array([min(r["val_loss"] * r["prepd_std"] ** 2 for r in t) for t in C.values()])
        done = np.array([len(t) >= 7 for t in C.values()])
        ax.plot(D, L, "-", color=col, label=name)
        ax.plot(D[done], L[done], "o", color=col, ls="none")
        ax.plot(D[~done], L[~done], "o", color=col, mfc="none", ls="none")
        print(p, arm, " ".join(f"{d:.0f}:{l:.2g}({len(t)})" for d, l, t in zip(D, L, C.values())))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.process_label(ax, lab)
    ps.legend(ax, "lower left")
ps.save_panels(figs, os.path.join(FIG, "transfer_ch1"))
