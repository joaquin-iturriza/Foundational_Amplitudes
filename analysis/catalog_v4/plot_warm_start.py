#!/usr/bin/env python3
"""plan warm_start: R over the parent (the het control, step 0) of the warm starts W_k and their matched controls C_k
against k / 4264, beside the change from scratch B (share 0.25, 4264 steps), on PP4 and EE4 (all classes), from
data/warm_start_<change>.json (mixture_ab.py warm). Points: R = median over classes of the class geometric mean of
per-probe ratios (seeds' geometric mean); bars the design's widened 95% class bootstrap.
-> figs/warm_start_<change>.png/.pdf"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter


def main(change="C1"):
    d = json.load(open(os.path.join(HERE, "data", f"warm_start_{change}.json")))
    ks = sorted(d["k"], key=int)
    x = [d["k"][k]["fraction"] for k in ks]
    fig, axes = ps.figure(ncols=2)
    for ax, r, lab in zip(axes, ("PP4", "EE4"), ("PP4", "EE4")):
        for name, col, mk, off, leg in (("W_over_A", ps.C.vermillion, "o", 1.03, "warm start, share 0.25"),
                                        ("C_over_A", ps.C.blue, "s", 0.97, "warm start, share 0")):
            ys = [d["k"][k][name][r]["all"]["R"] for k in ks]
            lo = [y - d["k"][k][name][r]["all"]["widened"][0] for k, y in zip(ks, ys)]
            hi = [d["k"][k][name][r]["all"]["widened"][1] - y for k, y in zip(ks, ys)]
            ax.errorbar([v * off for v in x], ys, yerr=[lo, hi], color=col, marker=mk, ls="-", label=leg)
        b = d["B_over_A"][r]["all"]["R"]
        ax.axhline(b, color=ps.C.black if hasattr(ps.C, "black") else "k", ls="--", label="from scratch, share 0.25")
        ax.axhline(1.0, color="0.6", ls=":", label="parent")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.5, 1.0, 2.0]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"$k\,/\,4264$")
        ax.set_ylabel(r"$R$ over the parent")
        ps.process_label(ax, lab)
    ps.legend(axes[1], "lower left", handles=axes[0].get_legend_handles_labels()[0])
    for ax in axes:
        ps.make_room(ax)
    ps.save(fig, os.path.join(HERE, "figs", f"warm_start_{change}"))


if __name__ == "__main__":
    main(*sys.argv[1:])
