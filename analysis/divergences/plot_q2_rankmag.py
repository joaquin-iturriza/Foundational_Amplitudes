#!/usr/bin/env python
"""Fig. q2_rankmag -- the controlled rank-vs-magnitude split of sigma-reweighting quality.

The order-only arms all reuse the SAME weight multiset Phi = sort|r| and differ only in which
event each weight is assigned to, so ordering is varied with magnitude held fixed. Comparing
them against the sigma head (which supplies both) separates the two contributions.

Left : fraction of oracle gain per arm, coloured by what the arm supplies.
Right: the same gain against ranking quality rho, with the real sigma head's two contributions
       marked at its own rho -- the ordering it achieves is worth much less than its magnitude.

Scored LOG-FLAT (mean of the per-y_min-decade MSEs), the metric Fig. q2_logflat establishes as
the honest one. This matters for what the figure says: event-weighted, ordering alone recovers
73% of the oracle gain and "magnitude dominates" is simply false; log-flat, the same arm
recovers 11%, because the worst-fit decade after antenna training is the BULK and only a
calibrated magnitude points there. The figure previously used the event-weighted numbers while
its caption quoted the log-flat ones, so the panel contradicted the claim above it.
CPU only.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

REPO = "/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps          # noqa: E402
import q2_metrics as q2          # noqa: E402

#: (tag, short tick name, family). The family sets the colour and is what the legend explains.
BARS = [
    ("sigma",        r"$\sigma$ head",        "both"),
    ("deg029",       r"degraded $\sigma$",     "both"),
    ("oracle",       r"oracle $|r|$",         "oracle"),
    ("rank_real",    r"order, $\sigma$",      "order"),
    ("rank_synth30", r"order, $\rho{=}0.30$", "order"),
    ("rank_synth46", r"order, $\rho{=}0.46$", "order"),
    ("rank_synth70", r"order, $\rho{=}0.70$", "order"),
]

#: Legend text names what each family actually provides to the reweighting proposal, so the
#: colours are self-explanatory without a note inside the axes.
FAMILY = {
    "both":   (ps.C.green,      "order and magnitude"),
    "oracle": (ps.C.blue,       "oracle"),
    "order":  (ps.C.vermillion, "order only"),
}

#: The order-only tolerance curve. rho=0 is the no-reweighting baseline and rho=1 the oracle
#: ordering, so the curve spans the whole reachable range rather than only the synthetic arms.
CURVE = ["baseQ", "rank_synth30", "rank_synth46", "rank_synth70", "oracle"]

#: Scored under the log-flat metric; see the module docstring for why not event-weighted.
METRIC = "logflat"


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--eval_dir", default=here)
    ap.add_argument("--out_base", default=os.path.join(here, "figs", "q2_rankmag"))
    ap.add_argument("--process", default=r"$e^+e^-\to u\bar u g$")
    args = ap.parse_args()

    tags = [t for t, _, _ in BARS] + CURVE
    G = q2.gains(tags, args.eval_dir)

    # TWO columns. The left panel's category names are long, and with the plot box
    # fixed those labels are charged to the canvas: side by side the figure needs ~7.1in. That
    # is over \textwidth but well inside MAX_FIG_IN, so the row simply overhangs the margins
    # (\widerow) and the two panels stay side by side, which is how they are read.
    fig, (axL, axR) = ps.figure(ncols=2)

    # --- left: gain by arm, coloured by what the arm supplies ------------------
    # HORIZONTAL bars: seven long category names ("order, $\rho{=}0.30$") needed
    # rotation=35 as vertical ticks, and a tall rotated label block used to be paid for out of
    # the plot. Sideways they read straight and cost only left margin.
    x = np.arange(len(BARS))
    for i, (t, _, fam) in enumerate(BARS):
        axL.barh(i, G[t][METRIC], 0.66, color=FAMILY[fam][0])
    axL.set_yticks(x)
    axL.set_yticklabels([nm for _, nm, _ in BARS])
    axL.invert_yaxis()
    axL.set_xlabel(r"$\dfrac{M_{\mathrm{base}}-M}{M_{\mathrm{base}}-M_{\mathrm{oracle}}}$")
    # Proxy handles: bars carry meaning through colour, so the legend must decode the colour.
    from matplotlib.patches import Patch
    seen, handles = set(), []
    for _, _, fam in BARS:
        if fam not in seen:
            seen.add(fam)
            handles.append(Patch(facecolor=FAMILY[fam][0], label=FAMILY[fam][1]))
    ps.legend(axL, "upper left", handles=handles)
    ps.process_label(axL, args.process, loc="lower right")

    # --- right: gain vs ranking quality, magnitude held fixed ------------------
    xs = [q2.RHO[t] for t in CURVE]
    ys = [G[t][METRIC] for t in CURVE]
    axR.plot(xs, ys, "o-", color=ps.C.vermillion, label="order only")
    axR.plot([q2.RHO["rank_real"]], [G["rank_real"][METRIC]], "*", ms=13,
             color=ps.C.green, label=r"$\sigma$, order only")
    axR.plot([q2.RHO["sigma"]], [G["sigma"][METRIC]], "P", ms=10,
             color=ps.C.blue, label=r"$\sigma$, order $+$ magnitude")
    axR.set_xlabel(r"ranking quality $\rho(\mathrm{score},|r|)$")
    axR.set_ylabel(r"$\dfrac{M_{\mathrm{base}}-M}{M_{\mathrm{base}}-M_{\mathrm{oracle}}}$")
    # Short labels: "order only, magnitude fixed" and "$\sigma$ head, order and magnitude"
    # made a legend wider than the plot box, which then hung over the panel beside it. What
    # is held fixed is already the point of the x-axis, and the caption says it in full.
    ps.legend(axR, "upper left")

    ps.save(fig, args.out_base)

    for t, nm, fam in BARS:
        print(f"{nm:26s} {fam:8s} gain={100 * G[t][METRIC]:6.1f}%")


if __name__ == "__main__":
    main()
