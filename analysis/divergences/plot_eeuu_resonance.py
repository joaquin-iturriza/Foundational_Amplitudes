#!/usr/bin/env python
"""ee->uubar Z-resonance: error vs sqrt(s) for the three sampling densities.

Rebuild of the eeuu_flatlogm_resonance figure. Two reasons it is rebuilt rather than reused:
  1. the original plotting script is not in the repo (lost);
  2. the old figure was drawn from eeuu_reson_summary.json, which is the PRE-FIX eval -- the
     in-process state leak across tags corrupted every tag after the first (`raw` was tag 1 and is
     byte-identical between the two summaries; flatlogm/genflat differ by 4-9x). We read the
     corrected re-eval, eeuu_reson_clean_summary.json, instead.

Three arms, all scored on the same RAMBO test set binned in sqrt(s):
  raw       flat RAMBO training density (starves the pole)
  flatlogm  flat-log|M|^2 obtained by RESAMPLING the fixed RAMBO pool (hits the coverage ceiling)
  genflat   flat-log|M|^2 obtained by GENERATING fresh events
CPU only. Emits .png and .pdf.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [("raw", "raw RAMBO", "#7F7F7F", "o", "-"),
        ("flatlogm", r"flat-$\log|\mathcal{M}|^2$ resampled", "#C44E52", "o", "-"),
        ("genflat", r"flat-$\log|\mathcal{M}|^2$ generated", "#4C72B0", "o", "-")]
XTICKS = [91, 100, 150, 300, 600, 1000]


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--summary", default=os.path.join(here, "eeuu_reson_clean_summary.json"))
    ap.add_argument("--out_base", default=os.path.join(here, "figs", "eeuu_flatlogm_resonance"))
    args = ap.parse_args()

    by_tag = {d["tag"]: d for d in json.load(open(args.summary))}

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for tag, label, colour, marker, ls in ARMS:
        if tag not in by_tag:
            continue
        b = np.array([[lo, hi, n, mse] for lo, hi, n, mse in by_tag[tag]["binmse"]], dtype=float)
        centre = np.sqrt(b[:, 0] * b[:, 1])          # geometric centre of each sqrt(s) bin
        ax.plot(centre, b[:, 3], marker + ls, color=colour, lw=1.8, ms=6, label=label)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([str(t) for t in XTICKS])
    ax.minorticks_off()
    ax.set_xlabel(r"$\sqrt{s}$  [GeV]")
    ax.set_ylabel(r"MSE  $\Delta\log|\mathcal{M}|^2$")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(fontsize=9, frameon=True)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out_base), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_base}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_base}.png/.pdf")

    r, g, f = by_tag["raw"], by_tag["genflat"], by_tag["flatlogm"]
    print(f"  raw->generated : logflat {r['logflat']:.3e} -> {g['logflat']:.3e} "
          f"({r['logflat']/g['logflat']:.1f}x)   Zpeak {r['zpeak']:.3e} -> {g['zpeak']:.3e} "
          f"({r['zpeak']/g['zpeak']:.1f}x)")
    print(f"  raw->resampled : logflat {r['logflat']:.3e} -> {f['logflat']:.3e} "
          f"({r['logflat']/f['logflat']:.1f}x)   Zpeak {r['zpeak']:.3e} -> {f['zpeak']:.3e} "
          f"({r['zpeak']/f['zpeak']:.1f}x)")
    print(f"  generated vs resampled (logflat): {f['logflat']/g['logflat']:.1f}x better")


if __name__ == "__main__":
    main()
