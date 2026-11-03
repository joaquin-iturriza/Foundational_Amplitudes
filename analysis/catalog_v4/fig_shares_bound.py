"""The default-sampled shares against the pure-default bound (docs/results.tex par:v4_mixture, tab:v4_bound_shares):
R (median over classes of the arm/control MSE ratio, widened 95% class-bootstrap interval) per read group, one panel per
training loss. MSE: pure default (the bound arm), share 0.50, share 0.25 against the MSE control pair
(data/bound_default.json); het: shares 0.50 and 0.25 against the het control pair (data/mixture_ab_het.json; no het
bound arm was trained). Every read on MadEvent default-sampled test events. Lines at R = 1 (the control) and R = 0.90
(the design's GAIN bar).

    python3 analysis/catalog_v4/fig_shares_bound.py  ->  analysis/catalog_v4/figs/shares_bound_{a,b}.{png,pdf}

With `arms`: the zero-shot reads per arm, each seed its own point (seed 42 filled, 43 open), no error bars
(data/arms_vs_het_control.json "seeds", mixture_ab.py arms), every arm at 4264 steps but the last, het loss, its epochs
(data/epochs.json) in its label. The arms: the control (share 0, width 8, 1e4 train events per process, in memory),
share 0.25 at width 8 on 1e4 events per process (step 15), and share 0.25 on the whole training set (1e5 per process,
memmap path) at widths 8, 16 and 32 (step 33), and the best trial of the 17056-step DyHPO sweep (width 8, whole set,
lr and HPs tuned, seed 42 only, not converged; step 16). Panel a: R per read (a seed's MSE over the control's two-seed geometric
mean, class median; the control's own seeds read the same way); b: the absolute MSE of log|M|^2 per read (class median
of the class's geometric mean over its probes); c and d: the same on the held-out pool by axis (default-sampled).

    python3 analysis/catalog_v4/fig_shares_bound.py arms  ->  analysis/catalog_v4/figs/arms_zero_shot_{a,b,c,d}.{png,pdf}
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


# the arms figure: reads, held-out axes in words, arms (label without its epochs, colour, marker)
ARM_READS = [("pp 13 TeV", "PP4"), (r"$e^+e^-$", "EE4"), ("held out, default-sampled", "held_default"),
             ("held out, catalog-sampled", "held"), ("held out, one-loop", "held_default/H5")]
ARM_AXES = [("H1", "new combinations"), ("H2", r"$2\to4$ families"), ("H2p5", r"$2\to5$"),
            ("H3", "physical top"), ("H6", r"$\alpha_s=0.13$"), ("H7E", r"$\sqrt{\hat s}$ 6 to 20 TeV"),
            ("H7c", r"$p_T\geq30$ GeV"), ("H5", "one loop")]
ARM_ARMS = [("control", r"control: share 0, width 8, $10^4$ events per process", "k", "s"),
            ("default25_1e4", r"share 0.25, width 8, $10^4$ events per process", ps.C.grey, "v"),
            ("w8_full", r"share 0.25, width 8, $10^5$ events per process", ps.C.blue, "o"),
            ("w16_full", r"share 0.25, width 16, $10^5$ events per process", ps.C.green, "D"),
            ("w32_full", r"share 0.25, width 32, $10^5$ events per process", ps.C.vermillion, "^"),
            ("long4x_best", r"share 0.25, width 8, $10^5$ events per process, 4$\times$ steps, tuned, one seed, "
             "not converged", ps.C.purple, "P")]


def seed_panel(ax, seeds, reads, arms, key, xlabel, ticks, ref=None):
    """One row per read, one marker per arm and seed (seed 42 filled, seed 43 open), no error bars"""
    from matplotlib.lines import Line2D
    n = len(reads)
    off = {a[0]: (i - (len(arms) - 1) / 2) * 0.17 for i, a in enumerate(arms)}
    for j, (lab, r) in enumerate(reads):
        for a, label, col, mk in arms:
            d = seeds[a][r]
            y = n - 1 - j + off[a]
            ax.plot([d["seed42"][key]], [y + 0.035], ls="none", marker=mk, color=col)
            if "seed43" in d:          # a one-seed arm (a sweep's best trial) has seed 42 only
                ax.plot([d["seed43"][key]], [y - 0.035], ls="none", marker=mk, color=col, mfc="none")
    if ref is not None:
        ax.axvline(ref[0], color=ps.C.grey, linestyle="--")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{lab} ({seeds[arms[0][0]][r]['n_probes']})" for lab, r in reads][::-1])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel(xlabel)
    hs = [Line2D([], [], ls="none", marker=mk, color=col, label=label) for _, label, col, mk in arms]
    hs += [Line2D([], [], ls="none", marker="o", color="k", label="seed 42"),
           Line2D([], [], ls="none", marker="o", color="k", mfc="none", label="seed 43")]
    if ref is not None:
        hs.append(Line2D([], [], color=ps.C.grey, ls="--", label=ref[1]))
    for h in hs:               # empty proxies, so the legend strip (ps.shared_legend reads the axes) names them
        ax.add_line(h)


def arms_figure():
    d = json.load(open(os.path.join(DATA, "arms_vs_het_control.json")))
    seeds, ep = d["seeds"], d["epochs"]
    eps = {a: sorted({round(v["epochs"], 2) for v in ep[a].values()}) for a in ep}
    assert all(len(v) == 1 for v in eps.values()), eps
    arms = [(a, f"{lab}, {eps[a][0]:.2g} epochs", c, m) for a, lab, c, m in ARM_ARMS]
    axes = [(lab, f"held_default/{ax}") for ax, lab in ARM_AXES]
    ref = (1.0, "control, geometric mean of its two seeds")
    rlab = r"$R$ = MSE(arm, seed) / MSE(control)"
    mlab = r"MSE of $\log|\mathcal{M}|^2$"
    figs = ps.panels(4)
    for (fig, ax), reads, key, xl, ticks, rf in (
            (figs[0], ARM_READS, "R", rlab, (0.1, 0.2, 0.5, 1.0), ref),
            (figs[1], ARM_READS, "mse", mlab, (0.1, 1.0, 10.0), None),
            (figs[2], axes, "R", rlab, (0.05, 0.1, 0.2, 0.5, 1.0), ref),
            (figs[3], axes, "mse", mlab, (0.001, 0.01, 0.1, 1.0, 10.0), None)):
        seed_panel(ax, seeds, reads, arms, key, xl, ticks, rf)
        ps.shared_legend(fig, ax, ncol=1)
    ps.save_panels(figs, os.path.join(HERE, "figs", "arms_zero_shot"))


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
