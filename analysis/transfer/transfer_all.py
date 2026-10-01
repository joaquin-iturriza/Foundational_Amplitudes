"""Transfer study, every probe fine-tuned from the factor-off ee->uu pretraining (lr_scale searched over [1,100])
against the same probe from scratch. The fine-tune keeps the pretraining's off-shellness input scale (tp3_ftp) for the
seven probes with an internal Z; the other five see only columns the pretraining saw constant, where tp3_ft is the
same run: the gain L_scratch / L_fine-tune per cell, each the best trial of an
8-trial single-fidelity DyHPO, MSE of log|M|^2 at the best checkpoint. The scratch arm is the one whose target is
the fine-tune's (checked cell by cell: equal prepd_std): tp3_scr for the probes the factors touch (the Z-window
probes and ee->WW), tp2_scr where only the t-channel factor reached the pool, tp_scr where neither did.
Prints the gain table, each probe's geometric mean over D, and the best trial's lr_scale.
<base>_loss_ee_a..f, <base>_loss_qcd_a..f: the same cells as losses, one panel per probe (groups a+b, c+d).
  <base>_a  ee->dd~, ee->nu_e nu_e~, ee->tt~, ee->WW      <base>_b  ee->dd~ and ee->bb~ at one loop
  <base>_c  ee->Za, ud->ud, uu~->gg                         <base>_d  uu~->Zg, Zgg, Zggg
A cell is one search, extended in place where its best sat at the top of its lr window; the rule, the scratch arm
per probe and ee->WW's steered pool from 10^2.5 are in cells.py.
Data: analysis/transfer/scratch_sweeps.json.
    python analysis/transfer/transfer_all.py
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ARM, scratch, finetune  # noqa: E402

LAB = {"ee_ddbar": r"$e^+e^-\to d\bar d$", "ee_nnbar": r"$e^+e^-\to\nu_e\bar\nu_e$", "ee_ttbar": r"$e^+e^-\to t\bar t$",
       "ee_WW": r"$e^+e^-\to W^+W^-$", "ee_dd_nlo": r"$e^+e^-\to d\bar d$ (1-loop)",
       "ee_bb_nlo": r"$e^+e^-\to b\bar b$ (1-loop)", "ee_Za": r"$e^+e^-\to Z\gamma$", "ud_ud": r"$ud\to ud$",
       "uubar_gg": r"$u\bar u\to gg$", "uubar_Zg": r"$u\bar u\to Zg$", "uubar_Zgg": r"$u\bar u\to Zgg$",
       "uubar_Zggg": r"$u\bar u\to Zggg$"}
GROUPS = [["ee_ddbar", "ee_nnbar", "ee_ttbar", "ee_WW"], ["ee_dd_nlo", "ee_bb_nlo"],
          ["ee_Za", "ud_ud", "uubar_gg"], ["uubar_Zg", "uubar_Zgg", "uubar_Zggg"]]


fig, axes = ps.figure(ncols=2, nrows=2)
for ax, group in zip(axes.flat, GROUPS):
    for p, col in zip(group, ps.CYCLE):
        D, G, scale = [], [], []
        for k in range(2, 9):
            (ls, _), (lf, bf) = scratch(p, k), finetune(p, k)
            if ls is None or lf is None:
                continue
            D.append(10 ** (k / 2)); G.append(ls / lf); scale.append((bf.get("fine_tune") or {}).get("lr_scale"))
        ax.plot(D, G, "o-", color=col, label=LAB[p])
        print(f"{p:11s} gain", " ".join(f"{g:5.2f}" for g in G), f"| geomean {np.exp(np.mean(np.log(G))):.2f}",
              "| lr_scale", " ".join(f"{s:.0f}" if s else "-" for s in scale))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"$L_{\rm scratch}/L_{\rm fine\text{-}tune}$")
    ax.set_ylim(0.05, 3e3)                    # one scale on every panel, so the gains compare across panels
    ps.legend(ax, "upper left")
SUF = ""
ps.save(fig, os.path.join(ROOT, "analysis", "transfer", "figs", "transfer_all" + SUF))


for part, order in (("ee", GROUPS[0] + GROUPS[1]), ("qcd", GROUPS[2] + GROUPS[3])):
  figs = ps.panels(len(order))
  for (fig, ax), p in zip(figs, order):
      for arm, col, name in (("scratch", ps.C.blue, "from scratch"), ("fine-tune", ps.C.vermillion, "fine-tuned")):
          D, L = [], []
          for k in range(2, 9):
              l, _ = scratch(p, k) if arm == "scratch" else finetune(p, k)
              if l is not None:
                  D.append(10 ** (k / 2)); L.append(l)
          ax.plot(D, L, "o-", color=col, label=name)
      ax.set_xscale("log"); ax.set_yscale("log")
      ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
      ps.process_label(ax, LAB[p])
      ps.legend(ax, "lower left")
      ps.make_room(ax)
  ps.save_panels(figs, os.path.join(ROOT, "analysis", "transfer", "figs", f"transfer_all_loss_{part}" + SUF))
