#!/usr/bin/env python
"""Generated mixture-fraction sweep for the ee->uu Z resonance.

f = fraction of training events drawn flat-log|M|^2, the rest uniform-sqrt(s).
  Left  : the three metrics vs f -> the interior optimum. The log-flat objective is DEFINED on the
          panel, since it is the metric everything else is judged against.
  Right : per-sqrt(s)-bin MSE coloured by f -> the bulk<->pole trade-off.

Reads the CORRECTED re-eval (eeuu_reson_clean_summary.json). Titles live in the figure caption, not
on the axes; no marker lines, and explicit sqrt(s) ticks. CPU only; emits .png and .pdf.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AN = os.path.dirname(os.path.abspath(__file__))
TAG2F = {"raw": 0.0, "mix025": 0.25, "mix050": 0.5, "mix075": 0.75, "genflat": 1.0}
XTICKS = [91, 100, 150, 300, 600, 1000]


def main():
    summ = json.load(open(os.path.join(AN, "eeuu_reson_clean_summary.json")))
    by = {d["tag"]: d for d in summ if d["tag"] in TAG2F}
    tags = sorted(by, key=lambda t: TAG2F[t])
    f = np.array([TAG2F[t] for t in tags])

    logflat = np.array([by[t]["logflat"] for t in tags])
    overall = np.array([by[t]["overall"] for t in tags])
    reson = np.array([np.nanmean(np.array(by[t]["binmse"], float)[
        np.array(by[t]["binmse"], float)[:, 0] < 100][:, 3]) for t in tags])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.6))

    # --- left: metrics vs f. Legend entries ARE the definitions (no bare parentheticals). ---
    axL.plot(f, logflat, "o-", color="#4c72b0", lw=2, ms=6,
             label=r"log-flat: mean of the per-$\sqrt{s}$-bin MSE")
    axL.plot(f, reson, "s-", color="#c44e52", lw=2, ms=6,
             label=r"resonance region: $\sqrt{s}<100$ GeV")
    axL.plot(f, overall, "^--", color="#7f7f7f", lw=1.6, ms=6,
             label="overall: plain mean over all events")
    axL.set_yscale("log")
    axL.set_xlabel(r"$f$ = fraction flat-$\log|\mathcal{M}|^2$   (rest uniform-$\sqrt{s}$)")
    axL.set_ylabel(r"MSE $\Delta\log|\mathcal{M}|^2$")
    axL.legend(frameon=False, fontsize=8.5, loc="upper center")
    axL.grid(True, which="both", alpha=0.2)

    # --- right: per-bin trade-off ---
    for t in tags:
        b = np.array(by[t]["binmse"], float)
        ctr = np.sqrt(b[:, 0] * b[:, 1])
        axR.plot(ctr, b[:, 3], "o-", color=plt.cm.viridis(TAG2F[t]), lw=1.8, ms=5,
                 label=f"f = {TAG2F[t]:g}")
    axR.set_xscale("log"); axR.set_yscale("log")
    axR.set_xticks(XTICKS); axR.set_xticklabels([str(t) for t in XTICKS]); axR.minorticks_off()
    axR.set_xlabel(r"$\sqrt{s}$ [GeV]")
    axR.set_ylabel(r"MSE $\Delta\log|\mathcal{M}|^2$")
    axR.legend(frameon=False, fontsize=8, ncol=2)
    axR.grid(True, which="major", alpha=0.2)

    fig.tight_layout()
    # Spell the objective out BELOW the axes -- inside the panel it would sit on top of the
    # "overall" curve. log-flat is the metric everything else is judged against, so define it.
    fig.subplots_adjust(bottom=0.30)
    fig.text(0.055, 0.045,
             r"objective:   $\mathrm{log\text{-}flat}\;=\;\frac{1}{N_{\mathrm{bins}}}"
             r"\sum_{b}\mathrm{MSE}_{b}$   —   every $\sqrt{s}$ bin counts once, "
             r"however many events it holds." "\n"
             r"The plain event mean instead follows the bulk, which holds almost all the events, "
             r"so it barely sees the pole.",
             fontsize=8.2, va="bottom", ha="left", color="#333333")

    base = os.path.join(AN, "eeuu_genmix_fraction_sweep")
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", base + ".png/.pdf")
    print(f"{'f':>5} {'logflat':>12} {'resonance':>12} {'overall':>12}")
    for i, t in enumerate(tags):
        print(f"{f[i]:5.2f} {logflat[i]:12.4e} {reson[i]:12.4e} {overall[i]:12.4e}")


if __name__ == "__main__":
    main()
