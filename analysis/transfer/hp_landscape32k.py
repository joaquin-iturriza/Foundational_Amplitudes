"""Transfer study, the 32k HP landscape at D = 10^3.5 (tp3_hpx32k_*, the user's call 2026-10-06): per probe, the loss
of every trial (MSE of log|M|^2 at the best checkpoint, val_loss * prepd_std^2) against its lr and its lambda, for
scratch, rung 0 (ee->uu) and rung 9. Filled: the wider DyHPO search (fine-tune lr [3e-4, 1e-2], lambda up to 1e-5);
open: the cell's two chosen points of the 32k grid (lr 1.2e-3, 1.8e-3 at lambda ~3e-10; scratch 7e-4, 1.4e-3 at 2e-7).
A trial that diverged has no result and is listed on stdout, not drawn.
    python analysis/transfer/hp_landscape32k.py      -> figs/hp_landscape32k_lr_*, figs/hp_landscape32k_lam_*
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, S  # noqa: E402
from rung_grid import LAB  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

PROBES = ["ee_ddbar", "ee_nnbar", "ee_dd_nlo", "ee_WW", "uubar_gg"]
FAM = {"scr": ("from scratch", "tp3_scr32k_{p}_d7", ps.C.black if hasattr(ps.C, "black") else "k"),
       "r0": ("rung 0", "tp3_uu64fte32k_{p}_d7", ps.C.blue),
       "r9": ("rung 9", "tp3_r9fte32k_{p}_d7", ps.C.vermillion)}
CHOSEN = json.load(open(os.path.join(ROOT, "analysis", "transfer", "horizon32k_chosen.json")))


def pts(name, hps=None):
    out = []
    for t in S.get(name, []):
        if t.get("val_loss") is None or not t.get("prepd_std") or (hps is not None and t["hp"] not in hps):
            continue
        out.append((t["lr"], (t.get("hps") or {}).get("lambda"), t["val_loss"] * t["prepd_std"] ** 2))
    return out


base = os.path.join(ROOT, "analysis", "transfer", "figs", "hp_landscape32k")
for key, xi, xlab in (("lr", 0, r"learning rate"), ("lam", 1, r"$\lambda$ (L2)")):
    figs = ps.panels(len(PROBES))
    for (fig, ax), p in zip(figs, PROBES):
        for f, (lab, pat, col) in FAM.items():
            grid = pat.format(p=p)
            new = pts(f"tp3_hpx32k_{f}_{p}_d7")
            old = pts(grid, set(CHOSEN.get(grid, [])))
            ax.scatter([q[xi] for q in new], [q[2] for q in new], color=col, label=f"{lab}, search")
            ax.scatter([q[xi] for q in old], [q[2] for q in old], facecolors="none", edgecolors=col,
                       label=f"{lab}, chosen points")
            print(f"{p:10s} {f:4s} search n={len(new)}  best {min((q[2] for q in new), default=float('nan')):.3g}"
                  f"  | chosen best {min((q[2] for q in old), default=float('nan')):.3g}")
        ax.set_xscale("log"); ax.set_yscale("log")
        for a_ in (ax.xaxis, ax.yaxis):          # decades only: minor labels collide on a sub-decade range
            a_.set_major_locator(LogLocator(base=10, numticks=6)); a_.set_minor_formatter(NullFormatter())
        ax.set_xlabel(xlab); ax.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
        ps.process_label(ax, LAB[p])
    ps.legend_strip(figs[0][1], base + f"_{key}_legend", ncol=3)
    ps.save_panels(figs, base + f"_{key}")
