"""The default-sampled shares against the pure-default bound (docs/results.tex par:v4_mixture, tab:v4_bound_shares):
R (median over classes of the arm/control MSE ratio, widened 95% class-bootstrap interval) per read group, one panel per
training loss. MSE: pure default (the bound arm), share 0.50, share 0.25 against the MSE control pair
(data/bound_default.json); het: shares 0.50 and 0.25 against the het control pair (data/mixture_ab_het.json; no het
bound arm was trained). Every read on MadEvent default-sampled test events. Lines at R = 1 (the control) and R = 0.90
(the design's GAIN bar).

    python3 analysis/catalog_v4/fig_shares_bound.py  ->  analysis/catalog_v4/figs/shares_bound.{png,pdf}
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import plot_style as ps  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter  # noqa: E402

DATA = os.path.join(HERE, "data")
# (label, path into an arm's result)
READS = [("PP4 trained", ("PP4", "trained")), ("PP4 not trained", ("PP4", "not_trained")), ("EE4", ("EE4", "all")),
         ("held out (700)", ("held_default", "pooled")), ("held out H3", ("held_default", "H3")),
         ("held out H5", ("held_default", "H5"))]
ARMS = [("bound", "pure default", ps.C.vermillion, "s"), ("default50", "share 0.50", ps.C.green, "o"),
        ("default25", "share 0.25", ps.C.blue, "D")]


def panel(ax, arms_res, present):
    n = len(READS)
    shown = [a for a in ARMS if a[0] in present]
    off = {a[0]: (i - (len(shown) - 1) / 2) * 0.22 for i, a in enumerate(shown)}
    for key, label, col, mk in shown:
        for j, (_, (r, part)) in enumerate(READS):
            d = arms_res[key][r][part]
            y = n - 1 - j + off[key]
            lo, hi = d["widened"]
            ax.errorbar([d["R"]], [y], xerr=[[d["R"] - lo], [hi - d["R"]]], color=col, marker=mk, linestyle="none",
                        label=label if j == 0 else None)
    ax.axvline(1.0, color=ps.C.grey, linestyle="--", label=r"$R=1$")
    ax.axvline(0.90, color=ps.C.grey, linestyle=":", label=r"$R=0.90$")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1.0, 2.0]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.2", "0.5", "1", "2"]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for lab, _ in READS][::-1])
    ax.set_xlabel(r"$R$ = arm / control")


def main():
    mse = json.load(open(os.path.join(DATA, "bound_default.json")))["arms"]
    het = json.load(open(os.path.join(DATA, "mixture_ab_het.json")))["shares"]
    fig, (a, b) = ps.figure(ncols=2)
    panel(a, mse, {"bound", "default50", "default25"})
    panel(b, het, {"default50", "default25"})
    ps.process_label(a, "MSE")
    ps.process_label(b, "het")
    ps.shared_legend(fig, a, ncol=5)    # one strip above: the dot rows fill every corner of both panels
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    ps.save(fig, os.path.join(HERE, "figs", "shares_bound"))


if __name__ == "__main__":
    main()
