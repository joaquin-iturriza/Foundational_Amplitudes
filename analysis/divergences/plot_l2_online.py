#!/usr/bin/env python
"""Plot the L2 online sigma-generation A/B: logflat trajectory per arm across rounds (the headline
static-vs-l2-vs-oracle comparison) + the per-round binmse profile for the l2 arm (does the worst
sqrt(s) region MIGRATE during training, and does l2 track it?). Reads the merged eval summary written
by eval_eeuu_resonance.py over tags <tag>_<arm>_mu_r<r>."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(REPO, "analysis/divergences/l2_online_summary.json"))
    ap.add_argument("--tag", default="run")
    ap.add_argument("--arms", default="static,l2,oracle")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(REPO, "analysis/divergences/l2_online"))
    args = ap.parse_args()

    recs = {d["tag"]: d for d in json.load(open(args.summary))}
    arms = args.arms.split(",")
    colors = {"static": "#555555", "l2": "#1f77b4", "oracle": "#2ca02c"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (1) logflat trajectory per arm
    ax = axes[0]
    for arm in arms:
        ys = []
        for r in range(args.rounds):
            t = f"{args.tag}_{arm}_mu_r{r}"
            ys.append(recs[t]["logflat"] if t in recs else np.nan)
        ax.plot(range(args.rounds), ys, "o-", color=colors.get(arm), label=arm, lw=2, ms=7)
    ax.set_yscale("log"); ax.set_xlabel("round"); ax.set_ylabel("logflat MSE (equal per sqrt(s) decade)")
    ax.set_title("A/B: logflat vs round"); ax.set_xticks(range(args.rounds))
    ax.legend(); ax.grid(alpha=0.3, which="both")

    # (2) zpeak (Z-peak bin MSE) trajectory
    ax = axes[1]
    for arm in arms:
        ys = [recs.get(f"{args.tag}_{arm}_mu_r{r}", {}).get("zpeak", np.nan) for r in range(args.rounds)]
        ax.plot(range(args.rounds), ys, "o-", color=colors.get(arm), label=arm, lw=2, ms=7)
    ax.set_yscale("log"); ax.set_xlabel("round"); ax.set_ylabel("Z-peak bin MSE")
    ax.set_title("pole fit vs round"); ax.set_xticks(range(args.rounds))
    ax.legend(); ax.grid(alpha=0.3, which="both")

    # (3) migration: l2 arm binmse profile per round (worst bin moving?)
    ax = axes[2]
    focus = "l2" if "l2" in arms else arms[0]
    cmap = plt.cm.viridis(np.linspace(0, 0.85, args.rounds))
    for r in range(args.rounds):
        t = f"{args.tag}_{focus}_mu_r{r}"
        if t not in recs:
            continue
        b = np.array(recs[t]["binmse"])            # rows [lo, hi, count, mse]
        centers = 0.5 * (b[:, 0] + b[:, 1])
        ax.plot(centers, b[:, 3], "o-", color=cmap[r], label=f"round {r}", lw=1.8, ms=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sqrt(s) [GeV]"); ax.set_ylabel("bin MSE")
    ax.set_title(f"error profile per round ({focus}) — migration"); ax.legend(); ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=130)
    fig.savefig(args.out + ".pdf")
    print(f"wrote {args.out}.png/.pdf")

    # text summary
    print("\n=== logflat by arm x round ===")
    for arm in arms:
        row = "  ".join(f"r{r}={recs.get(f'{args.tag}_{arm}_mu_r{r}',{}).get('logflat',float('nan')):.3e}"
                        for r in range(args.rounds))
        print(f"{arm:>7}: {row}")


if __name__ == "__main__":
    main()
