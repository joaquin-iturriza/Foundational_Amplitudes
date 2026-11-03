"""The one-loop sign extension on catalog_v4 (docs/results.tex par:v4_signext), drawn from data/signext_compare.json
(analysis/catalog_v4/signext_compare.py @9ec60f21; control: dimensionless target, Muon at the search's best HPs; arm: the
same with the sign extension's one-loop training pools; four seeds a side, 42 to 45; best checkpoints).

Panel a: per held-out sign-changing one-loop pool (21), the wrong-sign share of the extension against the control,
each the mean over the four seeds; the line is equality.
Panel b: per held-out axis, the extension over the control on the zero-shot MSE of log|M|^2 (median over probes of the
seed-mean ratio, filled), each seed's ratio (open), and the control's own seed-43-over-42 median ratio (cross).

    python3 analysis/catalog_v4/fig_signext.py  ->  analysis/catalog_v4/figs/signext_{a,b}.{png,pdf}
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter  # noqa: E402

AXES = [("H1", "new combinations"), ("H2", r"$2\to4$ families"), ("H2p5", r"$2\to5$"), ("H3", "physical top"),
        ("H6", r"$\alpha_s=0.13$"), ("H7E", r"$\sqrt{\hat s}$ 6 to 20 TeV"), ("H7c", r"$p_T\geq30$ GeV"),
        ("H5", "one loop")]


def main():
    d = json.load(open(os.path.join(HERE, "data", "signext_compare.json")))
    pools = d["per_pool_pct"]
    x = np.array([np.mean(v["ctrl"]) for v in pools.values()])
    y = np.array([np.mean(v["ext"]) for v in pools.values()])
    assert len(x) == d["pools"]
    assert int((y < x).sum()) == d["seed_mean"]["ext_lower"]

    figs = ps.panels(2)
    (fa, a), (fb, b) = figs
    lo, hi = 0.5, 60.0
    a.plot([lo, hi], [lo, hi], color=ps.C.grey, linestyle="--", label="equal")
    a.plot(x, y, ls="none", marker="o", color=ps.C.vermillion, label="one-loop pool")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(lo, hi)
    a.set_ylim(lo, hi)
    for ax_ in (a.xaxis, a.yaxis):
        ax_.set_major_locator(FixedLocator([1, 3, 10, 30]))
        ax_.set_major_formatter(FixedFormatter(["1", "3", "10", "30"]))
        ax_.set_minor_formatter(NullFormatter())
    a.set_xlabel(r"wrong sign, control (%)")
    a.set_ylabel(r"wrong sign, extension (%)")
    ps.legend(a, "upper left")

    m = d["mse_ratio"]
    n = len(AXES)
    for j, (k, _) in enumerate(AXES):
        r = m[k]
        yy = n - 1 - j
        b.plot(r["per_seed"], [yy] * len(r["per_seed"]), ls="none", marker="o", mfc="none", color=ps.C.blue,
               label="each seed" if j == 0 else None)
        b.plot([r["median_ext_over_ctrl"]], [yy], ls="none", marker="D", color=ps.C.vermillion,
               label="seed means" if j == 0 else None)
        b.plot([r["ctrl_seed43_over_42"]], [yy], ls="none", marker="x", color="k",
               label="control, seed 43 over 42" if j == 0 else None)
    b.axvline(1.0, color=ps.C.grey, linestyle="--", label="equal")
    b.set_xscale("log")
    ticks = [0.5, 0.7, 1.0, 1.4]
    b.xaxis.set_major_locator(FixedLocator(ticks))
    b.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    b.xaxis.set_minor_formatter(NullFormatter())
    b.set_xlim(0.45, 1.7)
    b.set_yticks(range(n))
    b.set_yticklabels([f"{lab} ({m[k]['probes']})" for k, lab in AXES][::-1])
    b.set_ylim(-0.6, n - 0.4 + 2.2)
    b.set_xlabel(r"MSE of $\log|\mathcal{M}|^2$, extension / control")
    ps.legend(b, "upper left")
    ps.save_panels(figs, os.path.join(HERE, "figs", "signext"))


if __name__ == "__main__":
    main()
