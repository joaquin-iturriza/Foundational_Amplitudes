#!/usr/bin/env python
"""Aggregate and plot the LLoCa / L-GATr / L-GATr-slim architecture comparison.

Each architecture is a 20-trial DyHPO sweep on the identical 25-process pretrain25
recipe data, matched to ~1.6M params. Two budget regimes share this script:

  * equal WALL-TIME  : t_steps set so every sweep costs the same wall-clock as the
                       LLoCa 15000-step baseline (lgatr 1151, slim 8184).
  * equal COMPUTE    : t_steps set so every sweep costs the same training FLOPs as
                       the LLoCa baseline (lgatr 732, slim 9850).

For each sweep we select the best trial by val_loss (the metric DyHPO optimises)
and report its paired test_loss (the honest held-out number for that model).

Usage:
    python tools/analyze_arch_comparison.py --regime walltime
    python tools/analyze_arch_comparison.py --regime compute
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"

# Per-step cost + size from the shared profiling run (compare_models/profile_flops):
# batchsize 1024, sparse block-diagonal xformers attention, V100.
PROFILE = {
    "LLoCa":        {"ms": 36.5,  "flops": 48_780_082_944,  "params": 1_607_060},
    "L-GATr full":  {"ms": 475.6, "flops": 999_198_112_160, "params": 1_635_736},
    "L-GATr-slim":  {"ms": 66.9,  "flops": 74_290_131_872,  "params": 1_569_765},
}

# t_steps per regime (LLoCa is always the 15000-step anchor).
REGIMES = {
    "walltime": {
        "LLoCa":       ("sweeps/pretrain25/pretrain25",             15000),
        "L-GATr full": ("sweeps/pretrain25_lgatr/pretrain25_lgatr", 1151),
        "L-GATr-slim": ("sweeps/pretrain25_slim/pretrain25_slim",   8184),
    },
    "compute": {
        "LLoCa":       ("sweeps/pretrain25/pretrain25",                       15000),
        "L-GATr full": ("sweeps/pretrain25_lgatr_eqc/pretrain25_lgatr_eqc",   732),
        "L-GATr-slim": ("sweeps/pretrain25_slim_eqc/pretrain25_slim_eqc",     9850),
    },
}

COLORS = {"LLoCa": "#1f77b4", "L-GATr full": "#d62728", "L-GATr-slim": "#2ca02c"}


def load_sweep(results_glob):
    """Return list of per-trial dicts with val_loss/test_loss/proc losses."""
    trials = []
    for f in sorted(glob.glob(results_glob)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("val_loss") is None:
            continue
        trials.append(d)
    return trials


def summarize(regime):
    rows = {}
    for name, (subdir, t_steps) in REGIMES[regime].items():
        trials = load_sweep(os.path.join(ROOT, subdir, "results", "*.json"))
        if not trials:
            print(f"  WARNING: no results yet for {name} ({subdir})")
            continue
        vals = np.array([t["val_loss"] for t in trials])
        best_i = int(np.argmin(vals))
        best = trials[best_i]
        rows[name] = {
            "n": len(trials),
            "t_steps": t_steps,
            "vals": vals,
            "best_val": float(vals[best_i]),
            "best_test": best.get("test_loss"),
            "median_val": float(np.median(vals)),
            "best_proc": best.get("proc_val_losses", {}),
            **PROFILE.get(name, {}),
        }
    return rows


def print_table(regime, rows):
    print(f"\n=== Architecture comparison — equal {regime.upper()} "
          f"(20-trial DyHPO each, ~1.6M params) ===")
    print(f"{'model':14s} {'steps':>6s} {'ms/step':>8s} {'best_val':>11s} "
          f"{'median_val':>11s} {'best_test':>11s}")
    for name, r in rows.items():
        bt = "   n/a" if r["best_test"] is None else f"{r['best_test']:.4e}"
        print(f"{name:14s} {r['t_steps']:6d} {r['ms']:8.1f} "
              f"{r['best_val']:.4e}  {r['median_val']:.4e}  {bt}")


def plot_distributions(regime, rows, outdir):
    """Strip + box of the 20-trial val_loss spread per architecture (log-y)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(rows.keys())
    for i, name in enumerate(names):
        v = rows[name]["vals"]
        x = np.full_like(v, i, dtype=float) + (np.random.RandomState(0).rand(len(v)) - 0.5) * 0.18
        ax.scatter(x, v, s=28, color=COLORS[name], alpha=0.7, zorder=3,
                   edgecolor="white", linewidth=0.4)
        ax.scatter([i], [rows[name]["best_val"]], marker="*", s=320,
                   color=COLORS[name], edgecolor="black", linewidth=0.8, zorder=4)
    bp = ax.boxplot([rows[n]["vals"] for n in names], positions=range(len(names)),
                    widths=0.5, showfliers=False, patch_artist=True, zorder=2)
    for patch in bp["boxes"]:
        patch.set(facecolor="none", edgecolor="grey")
    for med in bp["medians"]:
        med.set(color="grey")
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\n(t={rows[n]['t_steps']})" for n in names])
    ax.set_ylabel("val loss (combined, 25 processes)")
    ax.set_title(f"Equal {regime} — val-loss spread over 20 DyHPO trials\n"
                 "★ = best trial (selected model)")
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(outdir, f"arch_compare_{regime}_distribution.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_best_bars(regime, rows, outdir):
    """Best val and paired test loss per architecture (log-y bars)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(rows.keys())
    x = np.arange(len(names))
    w = 0.38
    val = [rows[n]["best_val"] for n in names]
    test = [rows[n]["best_test"] or np.nan for n in names]
    ax.bar(x - w / 2, val, w, label="best val loss",
           color=[COLORS[n] for n in names], alpha=0.95)
    ax.bar(x + w / 2, test, w, label="paired test loss",
           color=[COLORS[n] for n in names], alpha=0.5, hatch="//")
    for xi, v in zip(x - w / 2, val):
        ax.annotate(f"{v:.2e}", (xi, v), ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(t={rows[n]['t_steps']})" for n in names])
    ax.set_ylabel("loss")
    ax.set_title(f"Equal {regime} — best-trial loss per architecture")
    ax.legend()
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(outdir, f"arch_compare_{regime}_best_bars.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_efficiency(regime, rows, outdir):
    """Best val loss vs per-step cost (ms and FLOPs), annotated with step count."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, key, xlabel in [
        (axes[0], "ms", "per-step wall time (ms, batchsize 1024)"),
        (axes[1], "flops", "per-step training FLOPs"),
    ]:
        for name, r in rows.items():
            ax.scatter(r[key], r["best_val"], s=160, color=COLORS[name],
                       edgecolor="black", linewidth=0.8, zorder=3, label=name)
            ax.annotate(f"{name}\nt={r['t_steps']}", (r[key], r["best_val"]),
                        textcoords="offset points", xytext=(8, 6), fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("best val loss")
        ax.grid(which="both", alpha=0.25)
    axes[0].set_title(f"Equal {regime} — accuracy vs per-step cost")
    axes[1].set_title("(same, FLOPs axis)")
    fig.tight_layout()
    p = os.path.join(outdir, f"arch_compare_{regime}_efficiency.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["walltime", "compute"], default="walltime")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "compare_models", "figs"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = summarize(args.regime)
    print_table(args.regime, rows)
    if len(rows) < 3:
        print("\n(Not all sweeps have results yet — plots use what is available.)")
    if not rows:
        return
    paths = [
        plot_distributions(args.regime, rows, args.outdir),
        plot_best_bars(args.regime, rows, args.outdir),
        plot_efficiency(args.regime, rows, args.outdir),
    ]
    print("\nFigures written:")
    for p in paths:
        print("  " + p)


if __name__ == "__main__":
    main()
