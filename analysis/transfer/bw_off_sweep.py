"""Breit-Wigner factor off vs on, best vs best, for the four probes whose pool reaches the Z pole: each cell's best of an 8-trial search with
the factor off (tp3_scr, data.target_propagators false) against the best of the same search with it on
(tp_scr). Both are minima over eight trials, so the comparison is fair in selection (unlike the same-HP
shortcut, bw_off_shortcut.py). Loss is MSE of log|M|^2 at the best checkpoint; the factor is exact, so the
two arms share the unit. The ratio off/on is printed per cell.
Data: analysis/transfer/scratch_sweeps.json.
    python analysis/transfer/bw_off_sweep.py          -> figs/bw_off_sweep_a..d
  (a) ee->dd~, (b) ee->nu_e nu_e~, (c) ee->dd~ one loop, (d) ee->bb~ one loop
"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps

LAB = {"ee_ddbar": r"$e^+e^-\to d\bar d$", "ee_nnbar": r"$e^+e^-\to\nu_e\bar\nu_e$",
       "ee_dd_nlo": r"$e^+e^-\to d\bar d$ (1-loop)", "ee_bb_nlo": r"$e^+e^-\to b\bar b$ (1-loop)"}
S = json.load(open(os.path.join(ROOT, "analysis", "transfer", "scratch_sweeps.json")))


def best(fam, k, p):
    tr = S.get(f"{fam}_{p}_d{k}", [])          # the cell's own sweep only (a `_002` duplicate would add trials)
    tr = [t for t in tr if t.get("val_loss") is not None and t.get("prepd_std")]
    return min(t["val_loss"] * t["prepd_std"] ** 2 for t in tr) if tr else None


import numpy as np
PROBES = ["ee_ddbar", "ee_nnbar", "ee_dd_nlo", "ee_bb_nlo"]
figs = ps.panels(4)
for (fig, ax), p in zip(figs, PROBES):
    for fam, col, lab in [("tp_scr", ps.C.blue, "factor on"), ("tp3_scr", ps.C.vermillion, "factor off")]:
        pts = [(10 ** (k / 2), best(fam, k, p)) for k in range(2, 9)]
        pts = [(d, b) for d, b in pts if b is not None]
        ax.plot(*zip(*pts), "o-", color=col, label=lab)
    r = [best("tp3_scr", k, p) / best("tp_scr", k, p) for k in range(2, 9) if best("tp_scr", k, p) and best("tp3_scr", k, p)]
    print(p, "off/on", " ".join(f"{x:.2f}" for x in r), f"geomean {np.exp(np.mean(np.log(r))):.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"training events $D$"); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.legend(ax, "lower left")
    ps.process_label(ax, LAB[p])
    ps.make_room(ax)
ps.save_panels(figs, os.path.join(ROOT, "analysis", "transfer", "figs", "bw_off_sweep"))
