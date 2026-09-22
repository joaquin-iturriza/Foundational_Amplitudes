#!/usr/bin/env python
"""Generated mixture-fraction sweep for the ee->uu Z resonance.

f = fraction of training events drawn flat-log|M|^2, the rest uniform-sqrt(s).
  Left  : the three metrics vs f -> the interior optimum.
  Right : per-sqrt(s)-bin MSE coloured by f -> the bulk<->pole trade-off.

Reads the CORRECTED re-eval (eeuu_reson_clean_summary.json). What the metrics mean, and why
log-flat rather than the plain event mean, belongs in the results.tex caption, not on the axes.
CPU only; emits .png and .pdf.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

REPO = "/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import plot_style as ps  # noqa: E402

AN = os.path.dirname(os.path.abspath(__file__))
TAG2F = {"raw": 0.0, "mix025": 0.25, "mix050": 0.5, "mix075": 0.75, "genflat": 1.0}
XTICKS = [91, 150, 300, 600, 1000]


def main():
    summ = json.load(open(os.path.join(AN, "eeuu_reson_clean_summary.json")))
    by = {d["tag"]: d for d in summ if d["tag"] in TAG2F}
    tags = sorted(by, key=lambda t: TAG2F[t])
    f = np.array([TAG2F[t] for t in tags])

    logflat = np.array([by[t]["logflat"] for t in tags])
    overall = np.array([by[t]["overall"] for t in tags])
    reson = np.array([np.nanmean(np.array(by[t]["binmse"], float)[
        np.array(by[t]["binmse"], float)[:, 0] < 100][:, 3]) for t in tags])

    fig, (axL, axR) = ps.figure(ncols=2)

    axL.plot(f, logflat, "o-", color=ps.C.blue, label=r"log-flat over $\sqrt{s}$ bins")
    axL.plot(f, reson, "s-", color=ps.C.vermillion, label=r"$\sqrt{s}<100$ GeV")
    axL.plot(f, overall, "^--", color=ps.C.grey, label="mean over events")
    axL.set_yscale("log")
    axL.set_xlabel(r"$f$ = fraction flat-$\log|\mathcal{M}|^2$")
    axL.set_ylabel(r"MSE$(\Delta\log|\mathcal{M}|^2)$")
    axL.legend(loc="upper center")
    ps.process_label(axL, r"$e^+e^-\to u\bar u$", loc="lower right")

    ramp = ps.sequence(len(tags))
    for t, c in zip(tags, ramp):
        b = np.array(by[t]["binmse"], float)
        ctr = np.sqrt(b[:, 0] * b[:, 1])
        axR.plot(ctr, b[:, 3], "o-", color=c, label=f"$f={TAG2F[t]:g}$")
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xticks(XTICKS); axR.set_xticklabels([str(t) for t in XTICKS]); axR.minorticks_off()
    axR.set_xlabel(r"$\sqrt{s}$ [GeV]")
    axR.set_ylabel(r"MSE$(\Delta\log|\mathcal{M}|^2)$")
    axR.legend(loc="lower left", ncol=2)

    ps.save(fig, os.path.join(AN, "eeuu_genmix_fraction_sweep"))
    print(f"{'f':>5} {'logflat':>12} {'resonance':>12} {'overall':>12}")
    for i, t in enumerate(tags):
        print(f"{f[i]:5.2f} {logflat[i]:12.4e} {reson[i]:12.4e} {overall[i]:12.4e}")


if __name__ == "__main__":
    main()
