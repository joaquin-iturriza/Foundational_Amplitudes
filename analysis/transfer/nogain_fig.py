"""par:nogain's figure. (a) Over the 72 probe x D cells of the 8k grid, the median gain over the ten pretrainings
(log10 scratch / fine-tune) against scratch's drop into the cell from the half-decade before (data/nogain.json,
nogain.py). (b) ee -> Z gamma: the best trial's learning rate per D, scratch against the ten rungs' fine-tunes (median
of their best trials, and the range every fine-tune trial searched), from the sweeps' trial records (cells.py).
    .venv-driver/bin/python analysis/transfer/nogain_fig.py   -> figs/nogain.{png,pdf}
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cells import ARM, ROOT, S  # noqa: E402
from rung_grid import fam  # noqa: E402
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402

K = range(2, 9)


def main():
    d = json.load(open(os.path.join(HERE, "data", "nogain.json")))
    fig, (axL, axR) = ps.figure(ncols=2)
    x, y = [], []
    for p, v in d.items():
        for k, dr in v["scratch_drop"].items():
            g = [z for z in v["cells"][k]["gain"].values() if z is not None]
            x.append(dr); y.append(float(np.median(g)))
    axL.plot(x, y, "o", color=ps.C.blue, label=rf"72 cells, $r={np.corrcoef(x, y)[0, 1]:.2f}$")
    axL.set_xlabel(r"scratch's drop into the cell (decades)"); axL.set_ylabel(r"$\log_{10}$ (scratch / fine-tune)")
    ps.legend(axL, "upper right"); ps.make_room(axL)
    p = "ee_Za"
    Ds, sc, ft, lo, hi = [], [], [], [], []
    for k in K:
        st = [t for t in S.get(f"{ARM[p]}_{p}_d{k}", []) if t.get("val_loss")]
        fts = [[t for t in S.get(f"{fam(r)}_{p}_d{k}", []) if t.get("val_loss")] for r in range(10)]
        fts = [f for f in fts if f]
        if not st or not fts:
            continue
        Ds.append(10 ** (0.5 * k))
        sc.append(min(st, key=lambda t: t["val_loss"])["lr"])
        ft.append(float(np.median([min(f, key=lambda t: t["val_loss"])["lr"] for f in fts])))
        allf = [t["lr"] for f in fts for t in f]
        lo.append(min(allf)); hi.append(max(allf))
    axR.fill_between(Ds, lo, hi, color=ps.C.vermillion, alpha=0.2, label="fine-tune trials' range")
    axR.plot(Ds, ft, "o-", color=ps.C.vermillion, label="fine-tunes' best (median)")
    axR.plot(Ds, sc, "o-", color="k", label="scratch's best")
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xlabel(r"training events $D$"); axR.set_ylabel("learning rate")
    ps.process_label(axR, r"$e^+e^-\to Z\gamma$", loc="lower right")
    ps.legend(axR, "upper left"); ps.make_room(axR)
    ps.save(fig, os.path.join(HERE, "figs", "nogain"))


if __name__ == "__main__":
    main()
