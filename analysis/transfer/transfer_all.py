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
A cell with re-searches (the lr window moved up where the best sat at its top: scratch tp3_scrh/tp3_scrh10,
fine-tune tp3_ftph/tp3_fth) has several values; which one is reported is open for the user (docs/results.tex
sec:ladder-open), so the default is the original 8-trial search of each arm and the alternatives are flags:
  --select original   each arm's first search (default)
  --select latest     the last re-search where one exists (every cell one 8-trial search)
  --select union      the best over all of a cell's searches (8-24 trials, uneven between cells and arms)
Data: analysis/transfer/scratch_sweeps.json.
    python analysis/transfer/transfer_all.py [--select original|latest|union]
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps

_ap = argparse.ArgumentParser()
_ap.add_argument("--select", choices=["original", "latest", "union"], default="original")
SELECT = _ap.parse_args().select
S = json.load(open(os.path.join(ROOT, "analysis", "transfer", "scratch_sweeps.json")))
ARM = {"ee_ddbar": "tp3_scr", "ee_nnbar": "tp3_scr", "ee_dd_nlo": "tp3_scr", "ee_bb_nlo": "tp3_scr", "ee_WW": "tp3_scr",
       "ee_Za": "tp2_scr", "ud_ud": "tp2_scr", "uubar_gg": "tp2_scr", "uubar_Zg": "tp2_scr",
       "ee_ttbar": "tp_scr", "uubar_Zgg": "tp_scr", "uubar_Zggg": "tp_scr"}
LAB = {"ee_ddbar": r"$e^+e^-\to d\bar d$", "ee_nnbar": r"$e^+e^-\to\nu_e\bar\nu_e$", "ee_ttbar": r"$e^+e^-\to t\bar t$",
       "ee_WW": r"$e^+e^-\to W^+W^-$", "ee_dd_nlo": r"$e^+e^-\to d\bar d$ (1-loop)",
       "ee_bb_nlo": r"$e^+e^-\to b\bar b$ (1-loop)", "ee_Za": r"$e^+e^-\to Z\gamma$", "ud_ud": r"$ud\to ud$",
       "uubar_gg": r"$u\bar u\to gg$", "uubar_Zg": r"$u\bar u\to Zg$", "uubar_Zgg": r"$u\bar u\to Zgg$",
       "uubar_Zggg": r"$u\bar u\to Zggg$"}
GROUPS = [["ee_ddbar", "ee_nnbar", "ee_ttbar", "ee_WW"], ["ee_dd_nlo", "ee_bb_nlo"],
          ["ee_Za", "ud_ud", "uubar_gg"], ["uubar_Zg", "uubar_Zgg", "uubar_Zggg"]]


def scratch(p, k):
    """The cell's scratch value: the best over its search and, where its best sat at the lr window's top, the searches
    re-run half a decade and a decade higher (tp3_scrh, tp3_scrh10; same target, checked by prepd_std). 16-24
    trials against the fine-tune's 8, so the selection favours scratch: conservative for the gain."""
    return _pick([f"{ARM[p]}_{p}_d{k}", f"tp3_scrh_{p}_d{k}", f"tp3_scrh10_{p}_d{k}"])


def finetune(p, k):
    """The cell's fine-tune value, chosen as scratch's: the best over its search (tp3_ftp where the probe has an internal
    Z, else tp3_ft) and, where its best lr_scale sat at the top of [1, 100] (>= 10^1.7, the top 15% as for scratch),
    the re-search over [10, 1000] (tp3_ftph with the parent's off-shellness scale, tp3_fth for the probes without an
    internal Z, whose columns are all fitted on the probe as in tp3_ft)."""
    ft = f"tp3_ftp_{p}_d{k}" if f"tp3_ftp_{p}_d{k}" in S else f"tp3_ft_{p}_d{k}"
    return _pick([ft, f"tp3_ftph_{p}_d{k}" if f"tp3_ftp_{p}_d{k}" in S else f"tp3_fth_{p}_d{k}"])


def _pick(names):
    """One cell's value from its searches, original first, re-searches in order (--select); they must share the
    target (equal prepd_std)."""
    got = [best(n) for n in names]
    got = [g for g in got if g[0] is not None]
    stds = {round(g[1]["prepd_std"], 5) for g in got}
    assert len(stds) <= 1, f"{names}: searches on different targets (prepd_std {stds})"
    if not got:
        return None, None
    return {"original": got[0], "latest": got[-1], "union": min(got, key=lambda g: g[0])}[SELECT]


def best(name):
    tr = [t for t in S.get(name, []) if t.get("val_loss") is not None and t.get("prepd_std")]
    if not tr:
        return None, None
    b = min(tr, key=lambda t: t["val_loss"] * t["prepd_std"] ** 2)
    return b["val_loss"] * b["prepd_std"] ** 2, b


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
SUF = "" if SELECT == "original" else f"_{SELECT}"
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
