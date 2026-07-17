#!/usr/bin/env python
"""Recompute every Q2 arm under the LOG-FLAT-PER-DECADE metric (equal footing).

The eval summaries' 'mse' is np.mean(d^2) over ALL test events (d = Δlog|M|^2), which is
event-weighted -> dominated by the middle/deep y_min decades where the antenna test set
piles up (~50% of events below y_min 1e-3). That is NOT the project's equal-footing metric:
the whole point (geometric_mean loss, log-flat weighting) is to weight each y_min DECADE
equally so the multi-decade singular range counts the same everywhere -- exactly the metric
the RAMBO/antenna/mixture sampling study reported.

Here: per arm, from the saved per-event npz, compute
  mse_event : np.mean(d^2)                       (what was reported before)
  mse_decflat : mean over decades of mean(d^2)   (EQUAL footing per y_min decade)
and per-decade profiles, so we see the WHOLE range (bulk + deep IR), not just deep IR.
Re-derives gain-fraction-of-oracle under the equal-footing metric.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIV = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/analysis/divergences"
EDGES = 10.0 ** np.arange(-6, 1)          # 1e-6 .. 1e0 -> 6 decades
CEN = np.sqrt(EDGES[:-1] * EDGES[1:])


def metrics(npz):
    d = np.load(npz, allow_pickle=True)
    r2 = (np.asarray(d["true_logamp"], float) - np.asarray(d["pred_logamp"], float)) ** 2
    y = np.asarray(d["y_min"], float)
    b = np.clip(np.digitize(y, EDGES) - 1, 0, len(EDGES) - 2)
    per = np.array([r2[b == k].mean() if (b == k).any() else np.nan for k in range(len(EDGES) - 1)])
    return dict(mse_event=float(r2.mean()),
                mse_decflat=float(np.nanmean(per)),
                per_decade=per)


def main():
    arms = {}
    for f in glob.glob(os.path.join(DIV, "q2rw_eval_*.npz")):
        arms[os.path.basename(f)[len("q2rw_eval_"):-4]] = metrics(f)

    base = arms["baseQ"]; orc = arms["oracle"]
    def gf(m, key):
        return (base[key] - m[key]) / (base[key] - orc[key])

    show = ["baseQ", "deg029", "sigma", "oracle",
            "rank_synth30", "rank_synth46", "rank_synth70", "rank_real"]
    print(f"{'arm':>13} | {'event-wt MSE':>12} {'gain%':>6} | {'DECADE-FLAT MSE':>15} {'gain%':>6}")
    print("-" * 70)
    for a in show:
        if a not in arms:
            continue
        m = arms[a]
        print(f"{a:>13} | {m['mse_event']:12.4e} {100*gf(m,'mse_event'):5.0f}% | "
              f"{m['mse_decflat']:15.4e} {100*gf(m,'mse_decflat'):5.0f}%")

    # per-decade profile for the headline arms — does reweighting hurt the bulk?
    print(f"\nper-decade MSE (equal-footing view):")
    hdr = "  ".join(f"[1e{int(np.log10(EDGES[k])):d}]" for k in range(len(EDGES) - 1))
    print(f"{'arm':>10}  {hdr}")
    for a in ["baseQ", "sigma", "oracle"]:
        row = "  ".join(f"{v:7.1e}" for v in arms[a]["per_decade"])
        print(f"{a:>10}  {row}")

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(show))
    a0.bar(x - 0.2, [100 * gf(arms[a], "mse_event") for a in show], 0.4,
           label="event-weighted (old, deep-IR-dominated)", color="0.6")
    a0.bar(x + 0.2, [100 * gf(arms[a], "mse_decflat") for a in show], 0.4,
           label="decade-flat (equal footing, correct)", color="C0")
    a0.set_xticks(x); a0.set_xticklabels(show, rotation=30, ha="right", fontsize=8)
    a0.set_ylabel("% of oracle gain"); a0.axhline(0, color="k", lw=0.7)
    a0.set_title("Q2 gain: event-weighted vs equal-footing metric")
    a0.legend(fontsize=8); a0.grid(alpha=0.3, axis="y")

    for a, c in [("baseQ", "0.5"), ("deg029", "C3"), ("sigma", "C2"), ("oracle", "C0")]:
        a1.plot(CEN, arms[a]["per_decade"], "-o", ms=4, color=c, label=a)
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("y_min (decade centre)"); a1.set_ylabel("MSE Δlog|M|² in decade")
    a1.set_title("per-decade profile (whole range: bulk→deep IR)")
    a1.grid(alpha=0.3, which="both"); a1.legend(fontsize=8)
    fig.tight_layout()
    b = os.path.join(DIV, "figs", "q2_logflat")
    os.makedirs(os.path.dirname(b), exist_ok=True)
    fig.savefig(b + ".png", dpi=140); fig.savefig(b + ".pdf")
    print(f"\nwrote {b}.png/.pdf")


if __name__ == "__main__":
    main()
