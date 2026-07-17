#!/usr/bin/env python
"""Plot the L0 flat-log|M|^2 A/B: MSE(Δlog|M|^2) vs sqrt(s) for raw RAMBO vs flat-log|M|^2
resampling. The resampler used ONLY |M|^2; sqrt(s) here is the independent validation axis,
so a drop concentrated at the Z peak (sqrt(s)~91) proves structure-agnostic coverage.
Saves BOTH .png and .pdf."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
AN = os.path.join(REPO, "analysis/divergences")


def main():
    summ = json.load(open(os.path.join(AN, "eeuu_reson_summary.json")))
    by = {d["tag"]: d for d in summ}
    colors = {"raw": "#7f7f7f", "flatlogm": "#c44e52", "genflat": "#4c72b0"}
    labels = {"raw": "raw RAMBO (native density)",
              "flatlogm": r"flat-$\log|\mathcal{M}|^2$ RESAMPLED (coverage ceiling)",
              "genflat": r"flat-$\log|\mathcal{M}|^2$ GENERATED (L0)"}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for tag in ("raw", "flatlogm", "genflat"):
        if tag not in by:
            continue
        b = np.array(by[tag]["binmse"], float)     # lo, hi, n, mse
        ctr = np.sqrt(b[:, 0] * b[:, 1])            # geometric bin center (log axis)
        ax.plot(ctr, b[:, 3], "o-", color=colors[tag], label=labels[tag], lw=2, ms=6)
    ax.axvline(91.19, color="0.5", ls=":", lw=1)
    ax.text(91.19, ax.get_ylim()[1], r" $Z$ pole", va="top", ha="left", color="0.4", fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\sqrt{s}$ [GeV]  (validation axis --- unused by the generator's shaper)")
    ax.set_ylabel(r"MSE $\Delta\log|\mathcal{M}|^2$")
    r = by["raw"]
    title = "ee$\\to u\\bar u$ $Z$ resonance: generate, don't resample\n"
    if "genflat" in by:
        g = by["genflat"]
        title += (f"raw$\\to$gen: logflat {r['logflat']:.2g}$\\to${g['logflat']:.2g} "
                  f"({r['logflat']/g['logflat']:.1f}$\\times$), "
                  f"Zpeak {r['zpeak']:.2g}$\\to${g['zpeak']:.2g} "
                  f"({r['zpeak']/g['zpeak']:.1f}$\\times$)")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    base = os.path.join(AN, "eeuu_flatlogm_resonance")
    fig.savefig(base + ".png", dpi=150)
    fig.savefig(base + ".pdf")
    print("wrote", base + ".png/.pdf")


if __name__ == "__main__":
    main()
