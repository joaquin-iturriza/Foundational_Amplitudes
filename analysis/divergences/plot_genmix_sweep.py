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
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # repo root, so siteconf imports from anywhere
import siteconf

matplotlib.use("Agg")

REPO = siteconf.PROJECT_DIR
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

    # Joaquin 2026-11-17: the reader meets the sqrt(s) bins first (left), the fraction is a named colour bar, and the
    # f axis says it is a sampling technique (flat in log|M|^2 against flat phase space)
    fig, (axL, axR) = ps.figure(ncols=2)
    FLAB = "fraction of training events\n" + r"sampled flat in $\log|\mathcal{M}|^2$"
    cmap, norm = matplotlib.colormaps[ps.CMAP], matplotlib.colors.Normalize(0, 1)
    for t in tags:
        b = np.array(by[t]["binmse"], float)
        ctr = np.sqrt(b[:, 0] * b[:, 1])
        axL.plot(ctr, b[:, 3], "o-", color=cmap(norm(TAG2F[t])))
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xticks(XTICKS); axL.set_xticklabels([str(t) for t in XTICKS]); axL.minorticks_off()
    axL.set_xlabel(r"$\sqrt{\hat s}$ of the test events [GeV]")
    axL.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$ in the $\sqrt{\hat s}$ bin")
    ps.process_label(axL, r"$e^+e^-\to u\bar u$")
    ps.colorbar(axL, matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), FLAB + ",\nthe rest flat in phase space")

    axR.plot(f, logflat, "o-", color=ps.C.blue, label=r"mean over $\sqrt{\hat s}$ bins," + "\nequal weight per bin")
    axR.plot(f, reson, "s-", color=ps.C.vermillion, label=r"mean over bins with $\sqrt{\hat s}<100$ GeV")
    axR.plot(f, overall, "^--", color=ps.C.grey, label="mean over all test events")
    axR.set_yscale("log")
    axR.set_xlabel(FLAB)
    axR.set_ylabel(r"MSE$(\log|\mathcal{M}|^2)$")
    ps.legend(axR, "upper left")

    ps.save(fig, os.path.join(AN, "eeuu_genmix_fraction_sweep"))
    print(f"{'f':>5} {'logflat':>12} {'resonance':>12} {'overall':>12}")
    for i, t in enumerate(tags):
        print(f"{f[i]:5.2f} {logflat[i]:12.4e} {reson[i]:12.4e} {overall[i]:12.4e}")


if __name__ == "__main__":
    main()
