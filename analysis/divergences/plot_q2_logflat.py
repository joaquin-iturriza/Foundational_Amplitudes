#!/usr/bin/env python
"""Fig. q2_logflat -- the sigma-reweighting gain depends on how the eval weights y_min.

Left : fraction of the oracle reweighting gain for every arm, under the event-weighted metric
       and under the log-flat-per-decade metric. The two disagree, which is the point.
Right: per-decade MSE profile for the four calibrated arms, showing WHERE the disagreement
       comes from -- after antenna training the worst-fit decade is the bulk, not the deep IR.

Metrics and the gain definition: see q2_metrics.py. CPU only.
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

#: Left panel, in the order the argument is made: calibrated arms first, then order-only.
BARS = ["deg029", "sigma", "rank_synth30", "rank_synth46", "rank_synth70", "rank_real"]

#: Right panel: only the arms with a calibrated magnitude, else six curves overlap illegibly.
PROFILE = ["baseQ", "deg029", "sigma", "oracle"]

#: Short x-tick names. The full LABEL text is far too long for eight tick positions, and
#: shrinking the font to fit is not allowed -- so the bars get compact names and the legend
#: carries the metric distinction.
TICK = {
    "deg029":       r"degraded $\sigma$",
    "sigma":        r"$\sigma$ head",
    "rank_synth30": r"order, $\rho{=}0.30$",
    "rank_synth46": r"order, $\rho{=}0.46$",
    "rank_synth70": r"order, $\rho{=}0.70$",
    "rank_real":    r"order, $\sigma$",
}


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--eval_dir", default=here)
    ap.add_argument("--out_base", default=os.path.join(here, "figs", "q2_logflat"))
    ap.add_argument("--process", default=r"$e^+e^-\to u\bar u g$")
    args = ap.parse_args()

    G = q2.gains(BARS + PROFILE, args.eval_dir)

    fig, (axL, axR) = ps.figure(ncols=2)

    # --- left: the same arms scored by both metrics ---------------------------
    x = np.arange(len(BARS))
    w = 0.38
    ev = [G[t]["event"] for t in BARS]
    lf = [G[t]["logflat"] for t in BARS]
    axL.bar(x - w / 2, ev, w, color=ps.C.grey, label="event-weighted")
    axL.bar(x + w / 2, lf, w, color=ps.C.blue, label="log-flat per decade")
    axL.set_xticks(x)
    axL.set_xticklabels([TICK[t] for t in BARS], rotation=35, ha="right")
    # The formula IS the axis label: it says exactly how to read the scale (0 = no reweighting,
    # 1 = oracle) without a prose reading hint in the figure.
    axL.set_ylabel(r"$\dfrac{M_{\mathrm{base}}-M}{M_{\mathrm{base}}-M_{\mathrm{oracle}}}$")
    axL.axhline(0.0, color="black", lw=0.8, zorder=1)
    axL.legend(loc="upper left")
    ps.process_label(axL, args.process, loc="lower right")

    # --- right: where the two metrics part company ----------------------------
    ramp = dict(zip(PROFILE, [ps.C.grey, ps.C.vermillion, ps.C.green, ps.C.blue]))
    for t in PROFILE:
        axR.plot(q2.CENTRES, G[t]["metrics"]["per_decade"], "o-",
                 color=ramp[t], label=q2.LABEL[t])
    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set_xlabel(r"$y_{\min}$")
    # "in decade" dropped: the x-axis is y_min and the caption says the binning is per decade,
    # so it was redundant -- and it made the label 87% of the panel height, where savefig's
    # tight bbox sliced the last glyph off ("...in decad").
    axR.set_ylabel(r"MSE$(\Delta\log|\mathcal{M}|^2)$")
    axR.legend(loc="upper left")

    ps.save(fig, args.out_base)

    print(f"{'arm':30s} {'event-wt':>9s} {'log-flat':>9s}")
    for t in BARS:
        print(f"{q2.LABEL[t]:30s} {100 * G[t]['event']:8.1f}% {100 * G[t]['logflat']:8.1f}%")


if __name__ == "__main__":
    main()
