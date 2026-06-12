#!/usr/bin/env python3
"""
plot_fast_vs_original.py — val_loss vs wall-time, original (old code) vs the
`_fast` reruns (current optimized code), side by side.

For every cell that has a `<cell>_fast` counterpart it draws, per dataset size D
and per num_heads:
    - the original best point   (hollow marker, dashed curve)
    - the fast rerun point       (filled marker, solid curve)
    - an arrow original -> fast   (so the leftward = faster shift is obvious)

x-axis is measured wall-time (traintime_hours); y-axis is val_loss. Both log.

RUN ON JEAN ZAY (scans results across cells):
    python sweep/plot_fast_vs_original.py [--sweep-base DIR] [--suffix _fast] [--out PATH]
"""
import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# per-num_heads style (kept in sync with analyze_pretraining_scaling.NH_STYLE)
NH_STYLE = {
    2:  {"color": "crimson",      "marker": "v"},
    4:  {"color": "mediumpurple", "marker": "s"},
    8:  {"color": "seagreen",     "marker": "D"},
    16: {"color": "steelblue",    "marker": "o"},
    32: {"color": "darkorange",   "marker": "^"},
}
_DEFAULT_STYLE = {"color": "gray", "marker": "x"}


def _style(nh):
    return NH_STYLE.get(nh, _DEFAULT_STYLE)


def _d_value(d_key):
    """'1e4p5' -> 4.5  (the exponent), for ordering panels."""
    return float(d_key[2:].replace("p", "."))


def scan_best(results_dir):
    """(val_loss, traintime_hours) of the min-val_loss result JSON, or None."""
    if not os.path.isdir(results_dir):
        return None
    best = None
    for fn in os.listdir(results_dir):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(results_dir, fn)) as f:
                r = json.load(f)
        except Exception:
            continue
        vl = r.get("val_loss")
        if vl is None:
            continue
        tt = r.get("traintime_hours")
        if best is None or vl < best[0]:
            best = (float(vl), float(tt) if tt else float("nan"))
    return best


def main():
    ap = argparse.ArgumentParser(description="Plot original vs fast-rerun wall-time")
    ap.add_argument("--sweep-base",
                    default=os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "sweeps", "pretraining_scaling"))
    ap.add_argument("--suffix", default="_fast")
    ap.add_argument("--out", default=None, help="output PDF (default: <base>/fast_vs_original.pdf)")
    args = ap.parse_args()
    base = args.sweep_base
    out  = args.out or os.path.join(base, "fast_vs_original.pdf")

    names = set(os.listdir(base))
    # points[d_key][nh] = list of (t_steps, orig_val, orig_time, fast_val, fast_time)
    points = defaultdict(lambda: defaultdict(list))
    n_matched, n_pending = 0, 0
    for cell in sorted(names):
        if cell.endswith(args.suffix) or (cell + args.suffix) not in names:
            continue
        m_d  = re.search(r"_D(1e\d+(?:p\d+)?)", cell)
        m_nh = re.search(r"nh(\d+)", cell)
        m_t  = re.search(r"_t(\d+)", cell)
        if not (m_d and m_nh and m_t):
            continue
        d_key, nh, t = m_d.group(1), int(m_nh.group(1)), int(m_t.group(1))
        orig = scan_best(os.path.join(base, cell, "results"))
        fast = scan_best(os.path.join(base, cell + args.suffix, "results"))
        if orig is None:
            continue
        if fast is None:
            n_pending += 1
            continue
        points[d_key][nh].append((t, orig[0], orig[1], fast[0], fast[1]))
        n_matched += 1

    if not points:
        print(f"No matched original/{args.suffix} pairs with results yet "
              f"({n_pending} fast cells still pending). Nothing to plot.")
        return

    d_keys = sorted(points, key=_d_value)
    nD = len(d_keys)
    fig, axes = plt.subplots(1, nD, figsize=(4.2 * nD, 4.6), sharey=True, squeeze=False)
    axes = axes[0]

    speedups = defaultdict(list)   # nh -> [orig_time/fast_time]
    for ax, d_key in zip(axes, d_keys):
        for nh in sorted(points[d_key]):
            st = _style(nh)
            rows = sorted(points[d_key][nh])  # by t_steps
            ot = [r[2] for r in rows]; ov = [r[1] for r in rows]
            ft = [r[4] for r in rows]; fv = [r[3] for r in rows]
            # original: hollow + dashed; fast: filled + solid
            ax.plot(ot, ov, ls="--", color=st["color"], alpha=0.45, zorder=1)
            ax.scatter(ot, ov, marker=st["marker"], facecolors="none",
                       edgecolors=st["color"], s=45, zorder=3)
            ax.plot(ft, fv, ls="-", color=st["color"], alpha=0.9, zorder=2)
            ax.scatter(ft, fv, marker=st["marker"], color=st["color"], s=45, zorder=4)
            # arrows original -> fast
            for (_, ovl, oti, fvl, fti) in rows:
                if np.isfinite(oti) and np.isfinite(fti):
                    ax.annotate("", xy=(fti, fvl), xytext=(oti, ovl),
                                arrowprops=dict(arrowstyle="->", color=st["color"], alpha=0.5, lw=0.8))
                    if fti > 0:
                        speedups[nh].append(oti / fti)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"D={d_key}")
        ax.set_xlabel("wall-time  (traintime_hours)")
        ax.grid(True, which="both", alpha=0.15)
    axes[0].set_ylabel("val_loss")

    # legend: nh colors + hollow/filled meaning
    handles = [plt.Line2D([0], [0], marker=_style(nh)["marker"], color=_style(nh)["color"],
                          ls="", label=f"nh={nh}") for nh in sorted(NH_STYLE)
               if any(nh in points[d] for d in d_keys)]
    handles += [
        plt.Line2D([0], [0], marker="o", mfc="none", mec="k", ls="--", color="k", label="original"),
        plt.Line2D([0], [0], marker="o", color="k", ls="-", label="fast (optimized)"),
    ]
    axes[-1].legend(handles=handles, fontsize=8, loc="best")
    fig.suptitle("val_loss vs wall-time — original (hollow, dashed) → fast (filled, solid); "
                 "arrow = speedup", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=130, bbox_inches="tight")
    print(f"Wrote {out}  ({n_matched} matched pairs, {n_pending} fast cells still pending)")

    # median speedup per nh
    print("\nMedian wall-time speedup (original / fast):")
    for nh in sorted(speedups):
        s = speedups[nh]
        print(f"  nh={nh:<3d} {np.median(s):.2f}x   (n={len(s)})")


if __name__ == "__main__":
    main()
