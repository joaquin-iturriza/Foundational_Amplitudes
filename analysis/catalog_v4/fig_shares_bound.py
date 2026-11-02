"""The default-sampled shares against the pure-default bound (docs/results.tex par:v4_mixture, tab:v4_bound_shares):
R (median over classes of the arm/control MSE ratio, widened 95% class-bootstrap interval) per read group, one panel per
training loss. MSE: pure default (the bound arm), share 0.50, share 0.25 against the MSE control pair
(data/bound_default.json); het: shares 0.50 and 0.25 against the het control pair (data/mixture_ab_het.json; no het
bound arm was trained). Every read on MadEvent default-sampled test events. Lines at R = 1 (the control) and R = 0.90
(the design's GAIN bar).

    python3 analysis/catalog_v4/fig_shares_bound.py  ->  analysis/catalog_v4/figs/shares_bound_{a,b}.{png,pdf}

With `arms`: the zero-shot reads per arm against the het control (data/arms_vs_het_control.json, mixture_ab.py arms),
every arm at 4264 steps, het loss. The control (R = 1): share 0, width 8, 1e4 train events per process (in memory);
the arms: share 0.25 at width 8 on 1e4 events per process (step 15), and share 0.25 on the whole training set (1e5
per process, memmap path) at widths 8, 16 and 32 (step 33). Panel a the reads, panel b the held-out pool by axis
(default-sampled events).

    python3 analysis/catalog_v4/fig_shares_bound.py arms  ->  analysis/catalog_v4/figs/arms_zero_shot_{a,b}.{png,pdf}
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
# labels in words: pp 13 TeV MadEvent probes of the trained / untrained classes, the e+e- probes, the held-out pool,
# its physical-top classes and its one-loop classes; class counts from data/bound_default.json
READS = [("pp 13 TeV, trained classes (797)", ("PP4", "trained")),
         ("pp 13 TeV, untrained classes (504)", ("PP4", "not_trained")), (r"$e^+e^-$ (169)", ("EE4", "all")),
         ("held out, default-sampled (700)", ("held_default", "pooled")),
         ("held out, physical top (279)", ("held_default", "H3")), ("held out, one-loop (28)", ("held_default", "H5"))]
ARMS = [("bound", "pure default", ps.C.vermillion, "s"), ("default50", "share 0.50", ps.C.green, "o"),
        ("default25", "share 0.25", ps.C.blue, "D")]


# the arms figure: reads, held-out axes in words, arms
ARM_READS = [("pp 13 TeV", ("PP4", "all")), (r"$e^+e^-$", ("EE4", "all")),
             ("held out, default-sampled", ("held_default", "pooled")),
             ("held out, catalog-sampled", ("held", "pooled")), ("held out, one-loop", ("held_default", "H5"))]
ARM_AXES = [("H1", "new combinations"), ("H2", r"$2\to4$ families"), ("H2p5", r"$2\to5$"),
            ("H3", "physical top"), ("H6", r"$\alpha_s=0.13$"), ("H7E", r"$\sqrt{\hat s}$ 6 to 20 TeV"),
            ("H7c", r"$p_T\geq30$ GeV"), ("H5", "one loop")]
ARM_ARMS = [("default25_1e4", r"share 0.25, width 8, $10^4$/process", ps.C.grey, "s"),
            ("w8_full", r"share 0.25, width 8, $10^5$/process", ps.C.blue, "o"),
            ("w16_full", r"share 0.25, width 16, $10^5$/process", ps.C.green, "D"),
            ("w32_full", r"share 0.25, width 32, $10^5$/process", ps.C.vermillion, "^")]


def panel(ax, arms_res, present, reads=READS, arms=ARMS, lines=((1.0, "--", r"$R=1$"), (0.90, ":", r"$R=0.90$")),
          ticks=(0.2, 0.5, 1.0, 2.0), counts=False, xlabel=r"$R$ = arm / control"):
    n = len(reads)
    shown = [a for a in arms if a[0] in present]
    off = {a[0]: (i - (len(shown) - 1) / 2) * (0.22 if len(shown) < 4 else 0.18) for i, a in enumerate(shown)}
    labels = []
    for j, (lab, (r, part)) in enumerate(reads):
        labels.append(f"{lab} ({arms_res[shown[0][0]][r][part]['n_classes']})" if counts else lab)
        for key, label, col, mk in shown:
            d = arms_res[key][r][part]
            y = n - 1 - j + off[key]
            lo, hi = d["widened"]
            ax.errorbar([d["R"]], [y], xerr=[[d["R"] - lo], [hi - d["R"]]], color=col, marker=mk, linestyle="none",
                        label=label if j == 0 else None)
    for x, ls, lab in lines:
        ax.axvline(x, color=ps.C.grey, linestyle=ls, label=lab)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel(xlabel)


def arms_figure():
    res = json.load(open(os.path.join(DATA, "arms_vs_het_control.json")))["arms"]
    kw = dict(arms=ARM_ARMS, lines=((1.0, "--", r"control: share 0, width 8, $10^4$/process"),),
              ticks=(0.1, 0.2, 0.5, 1.0), counts=True, xlabel=r"$R$ = MSE(arm) / MSE(control)")
    (fa, a), (fb, b) = ps.panels(2)
    panel(a, res, set(res), reads=ARM_READS, **kw)
    panel(b, res, set(res), reads=[(lab, ("held_default", ax)) for ax, lab in ARM_AXES], **kw)
    ps.shared_legend(fa, a, ncol=1)
    ps.save_panels([(fa, a), (fb, b)], os.path.join(HERE, "figs", "arms_zero_shot"))


def main():
    mse = json.load(open(os.path.join(DATA, "bound_default.json")))["arms"]
    het = json.load(open(os.path.join(DATA, "mixture_ab_het.json")))["shares"]
    # one file per loss, stacked: the read groups' labels in words make a two-across row wider than the paper
    (fa, a), (fb, b) = ps.panels(2)
    panel(a, mse, {"bound", "default50", "default25"})
    panel(b, het, {"default50", "default25"})
    ps.process_label(a, "MSE")
    ps.process_label(b, "het")
    ps.shared_legend(fa, a, ncol=3)     # one strip above the first panel: the dot rows fill every corner
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    ps.save_panels([(fa, a), (fb, b)], os.path.join(HERE, "figs", "shares_bound"))

if __name__ == "__main__":
    arms_figure() if sys.argv[1:] == ["arms"] else main()
