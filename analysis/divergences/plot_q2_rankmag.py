#!/usr/bin/env python
"""Fig. q2_rankmag -- the controlled rank-vs-magnitude split of sigma-reweighting quality.

The order-only arms all reuse the SAME weight multiset Phi = sort|r| and differ only in which
event each weight is assigned to, so ordering is varied with magnitude held fixed. Comparing
them against the sigma head (which supplies both) separates the two contributions.

Left : fraction of oracle gain per arm, coloured by what the arm supplies.
Right: the same gain against ranking quality rho, with the real sigma head's two contributions
       marked at its own rho -- the ordering it achieves is worth much less than its magnitude.

Scored event-weighted, the metric under which the rank-vs-magnitude question was originally
posed; Fig. q2_logflat is the companion showing that the log-flat metric reverses the ranking.
CPU only.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style as ps          # noqa: E402
import q2_metrics as q2          # noqa: E402

#: (tag, short tick name, family). The family sets the colour and is what the legend explains.
BARS = [
    ("sigma",        r"$\sigma$ head",        "both"),
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
    "order":  (ps.C.vermillion, "order only, magnitude fixed"),
}

#: The order-only tolerance curve. rho=0 is the no-reweighting baseline and rho=1 the oracle
#: ordering, so the curve spans the whole reachable range rather than only the synthetic arms.
CURVE = ["baseQ", "rank_synth30", "rank_synth46", "rank_synth70", "oracle"]


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--eval_dir", default=here)
    ap.add_argument("--out_base", default=os.path.join(here, "figs", "q2_rankmag"))
    ap.add_argument("--process", default=r"$e^+e^-\to u\bar u g$")
    args = ap.parse_args()

    tags = [t for t, _, _ in BARS] + CURVE
    G = q2.gains(tags, args.eval_dir)

    fig, (axL, axR) = ps.figure(ncols=2)

    # --- left: gain by arm, coloured by what the arm supplies ------------------
    x = np.arange(len(BARS))
    for i, (t, _, fam) in enumerate(BARS):
        axL.bar(i, G[t]["event"], 0.66, color=FAMILY[fam][0])
    axL.set_xticks(x)
    axL.set_xticklabels([nm for _, nm, _ in BARS], rotation=35, ha="right")
    axL.set_ylabel(r"$\dfrac{M_{\mathrm{base}}-M}{M_{\mathrm{base}}-M_{\mathrm{oracle}}}$")
    # Proxy handles: bars carry meaning through colour, so the legend must decode the colour.
    from matplotlib.patches import Patch
    seen, handles = set(), []
    for _, _, fam in BARS:
        if fam not in seen:
            seen.add(fam)
            handles.append(Patch(facecolor=FAMILY[fam][0], label=FAMILY[fam][1]))
    axL.legend(handles=handles, loc="upper left")
    ps.process_label(axL, args.process, loc="lower right")

    # --- right: gain vs ranking quality, magnitude held fixed ------------------
    xs = [q2.RHO[t] for t in CURVE]
    ys = [G[t]["event"] for t in CURVE]
    axR.plot(xs, ys, "o-", color=ps.C.vermillion, label="order only, magnitude fixed")
    axR.plot([q2.RHO["rank_real"]], [G["rank_real"]["event"]], "*", ms=13,
             color=ps.C.green, label=r"$\sigma$ head, order only")
    axR.plot([q2.RHO["sigma"]], [G["sigma"]["event"]], "P", ms=10,
             color=ps.C.blue, label=r"$\sigma$ head, order and magnitude")
    axR.set_xlabel(r"ranking quality $\rho(\mathrm{score},|r|)$")
    axR.set_ylabel(r"$\dfrac{M_{\mathrm{base}}-M}{M_{\mathrm{base}}-M_{\mathrm{oracle}}}$")
    axR.legend(loc="upper left")

    ps.save(fig, args.out_base)

    for t, nm, fam in BARS:
        print(f"{nm:26s} {fam:8s} gain={100 * G[t]['event']:6.1f}%")


if __name__ == "__main__":
    main()
