#!/usr/bin/env python3
"""plan warm_start: R over the parent (the het control, step 0) of the warm starts W_k and their matched controls C_k
against k / 4264, beside the change from scratch B (share 0.25, 4264 steps), on PP4 and EE4 (all classes), from
data/warm_start_<change>.json (mixture_ab.py warm). Points: R = median over classes of the class geometric mean of
per-probe ratios (seeds' geometric mean); bars the design's widened 95% class bootstrap.
-> figs/warm_start_<change>.png/.pdf

  changes   step 7 (D283, D300): R(W_k/B) of every change against k / 4264 on its deciding reads, one panel per read:
            (a) PP4 (C1-C4 all classes, C5 the classes it trains on pp events), (b) EE4 (C1-C4), (c) the one-loop
            changes: C8 on HD5 (zero-shot MSE, as (a), (b)) and C6 the wrong-sign share's ratio of medians over the
            21 sign-changing held-out pools (signext_compare.py warm); bars the designs' widened 95% intervals, the
            dashed line the MATCH bar 1.35, the dotted line equality; series offset in k for legibility
            -> figs/warm_start_changes_{a,b,c,legend}.png/.pdf"""
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


# (change, label, colour, marker) of step 7's figure; C1 is the earlier result each change is set against
CHANGES = (("C1", "het loss, share 0.25", ps.C.grey, "o"), ("C2", "het loss, share 0.50", ps.C.vermillion, "s"),
           ("C3", "MSE loss, share 0.25", ps.C.blue, "D"), ("C4", "MSE loss, share 0.50", ps.C.green, "^"),
           ("C5", "MSE loss, pp-mixed set", ps.C.orange, "v"))


def changes():
    figs = ps.panels(3)
    load = lambda c: json.load(open(os.path.join(HERE, "data", f"warm_start_{c}.json")))
    series = {"a": [(c, lab, col, mk, ("PP4", "trained" if c == "C5" else "all")) for c, lab, col, mk in CHANGES],
              "b": [(c, lab, col, mk, ("EE4", "all")) for c, lab, col, mk in CHANGES if c != "C5"],
              "c": [("C8", "het, magnitude target", ps.C.purple, "o", ("HD5", "all")),
                    ("C6", "MSE, sign extension", ps.C.sky, "s", None)]}
    labels = {"a": "pp probes", "b": r"$e^+e^-$ probes", "c": "one-loop held-out pools"}
    for (fig, ax), key in zip(figs, "abc"):
        n = len(series[key])
        for i, (c, lab, col, mk, read) in enumerate(series[key]):
            d = load(c)
            ks = sorted(d["k"], key=int)
            off = 1.04 ** (i - (n - 1) / 2)
            x = [int(k) / 4264 * off for k in ks]
            if read is None:     # C6: ratio of medians of the wrong-sign share
                rows = [d["k"][k]["W_over_B"] for k in ks]
                ys, wid = [r["R"] for r in rows], [r["widened"] for r in rows]
            else:
                rows = [d["k"][k]["W_over_B"][read[0]][read[1]] for k in ks]
                ys, wid = [r["R"] for r in rows], [r["widened"] for r in rows]
            ax.errorbar(x, ys, yerr=[[y - w[0] for y, w in zip(ys, wid)], [w[1] - y for y, w in zip(ys, wid)]],
                        color=col, marker=mk, ls="-", label=lab)
        ax.axhline(1.35, color="0.3", ls="--", label="equivalence bar 1.35")
        ax.axhline(1.0, color="0.6", ls=":", label="equal to from scratch")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([0.5, 0.7, 1.0, 1.35, 2.0]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(FixedLocator([0.125, 0.25, 0.5]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"$k\,/\,4264$")
        ax.set_ylabel(r"$R$, warm start over from scratch")
        ax.set_ylim(0.35, 2.6)
        ps.process_label(ax, labels[key], loc="lower right")
        if key == "c":
            h, l = ax.get_legend_handles_labels()
            ps.legend(ax, "upper left", handles=h[2:], labels=l[2:])
        ps.make_room(ax)
    base = os.path.join(HERE, "figs", "warm_start_changes")
    ps.legend_strip(figs[0][1], base + "_legend", ncol=4)
    ps.save_panels(figs, base)


if __name__ == "__main__":
    changes() if sys.argv[1:] == ["changes"] else main(*sys.argv[1:])
