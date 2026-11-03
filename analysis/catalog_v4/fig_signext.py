"""Zero-shot error on the held-out one-loop processes with and without more one-loop training pools (docs/results.tex
par:v4_signext): the control (dimensionless target, Muon at the search's best HPs, catalog_v4's training set) and the
same with the sign extension's extra one-loop training pools (recipes/catalog_v4_train_signext.yaml), seeds 42 to 45,
best checkpoints. Per held-out one-loop process (axis H5, 28), the MSE of log|M|^2 (magnitude: mean (q - t)^2 of the
saved per-event reads, signext_compare.mse), each arm the mean over its four seeds; the line is equality.

    python3 analysis/catalog_v4/fig_signext.py  ->  analysis/catalog_v4/figs/loop_more_pools.{png,pdf}
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import plot_style as ps  # noqa: E402
import signext_compare as sc  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter  # noqa: E402


def main():
    m = {a: sc.mse(a, "H5") for a in sc.CTRL + sc.EXT}
    probes = sorted(set.intersection(*(set(v) for v in m.values())))
    x = np.array([[m[a][p] for p in probes] for a in sc.CTRL]).mean(0)
    y = np.array([[m[a][p] for p in probes] for a in sc.EXT]).mean(0)
    ref = json.load(open(os.path.join(HERE, "data", "signext_compare.json")))["mse_ratio"]["H5"]
    assert len(probes) == ref["probes"] and round(float(np.median(y / x)), 3) == ref["median_ext_over_ctrl"]
    print(json.dumps({"processes": len(probes), "lower": int((y < x).sum()),
                      "median_ratio": round(float(np.median(y / x)), 3),
                      "median_mse_control": round(float(np.median(x)), 4), "median_mse_more": round(float(np.median(y)), 4)}))

    fig, ax = ps.figure()
    lo, hi = 0.25 * min(x.min(), y.min()), 1.25 * max(x.max(), y.max())   # room for the legend below the points
    ax.plot([lo, hi], [lo, hi], color=ps.C.grey, linestyle="--", label="equal")
    ax.plot(x, y, ls="none", marker="o", color=ps.C.vermillion, label=f"one-loop process ({len(probes)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [t for t in (0.03, 0.1, 0.3, 1, 3, 10, 30) if lo <= t <= hi]
    for a_ in (ax.xaxis, ax.yaxis):
        a_.set_major_locator(FixedLocator(ticks))
        a_.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
        a_.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"MSE of $\log|\mathcal{M}|^2$, control")
    ax.set_ylabel(r"same, more one-loop training")
    ps.legend(ax, "lower right")
    ps.save(fig, os.path.join(HERE, "figs", "loop_more_pools"))


if __name__ == "__main__":
    main()
