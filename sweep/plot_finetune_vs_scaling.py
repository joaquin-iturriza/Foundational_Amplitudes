#!/usr/bin/env python3
"""
plot_finetune_vs_scaling.py  —  Compare fine-tune scaling to solo-from-scratch scaling.

For each dataset, plots:
  • Blue circles : solo scratch best per cell (from scaling_solo_NNN sweep)
  • Orange stars : fine-tune best per fidelity level (from finetune_scaling sweep)
  • Power-law fit on the fine-tune points

Usage:
    python sweep/plot_finetune_vs_scaling.py \\
        --finetune-config sweeps/finetune_scaling_eeuunlovirt_005/sweep_config.yaml \\
        --solo-dir        sweeps/scaling_solo_007_eeuunlovirte4 \\
        --out             sweeps/finetune_vs_solo_nlovirt_eeuu.pdf

    # For multiple datasets, pass multiple --solo-dir entries matching dataset order:
    python sweep/plot_finetune_vs_scaling.py \\
        --finetune-config sweeps/finetune_scaling_eeuunlovirt_005/sweep_config.yaml \\
        --solo-dir        sweeps/scaling_solo_007_eeuunlovirte4 \\
        --finetune-config sweeps/finetune_scaling_eettbarnlovirt_005/sweep_config.yaml \\
        --solo-dir        sweeps/scaling_solo_008_eettbarnlovirte4 \\
        --out             sweeps/finetune_vs_solo_nlovirt.pdf
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ds_tag(name):
    return (name.replace('_amplitudes', '').replace(', ', '_').replace('_', ''))


def _sweep_dirs(cfg, sweep_name):
    if "sweep_dir" in cfg.get("paths", {}):
        d = os.path.join(cfg["paths"]["sweep_dir"], sweep_name)
        return d, d
    afs = os.path.join(cfg["paths"]["afs_sweep_dir"], sweep_name)
    eos = os.path.join(cfg["paths"]["eos_sweep_dir"], sweep_name)
    return afs, eos


def load_results_from_dir(results_dir):
    results = []
    if not os.path.isdir(results_dir):
        return results
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(results_dir, fname)) as f:
                r = json.load(f)
            results.append(r)
        except Exception:
            continue
    return results


def fit_power_law(computes, losses):
    log_c = np.array([math.log(c) for c in computes])
    log_v = np.array([math.log(v) for v in losses])
    X = np.column_stack([np.ones_like(log_c), log_c])
    coeffs, _, _, _ = np.linalg.lstsq(X, log_v, rcond=None)
    log_A, neg_alpha = coeffs
    A     = math.exp(log_A)
    alpha = -neg_alpha
    log_v_pred = log_A + neg_alpha * log_c
    ss_res = np.sum((log_v - log_v_pred) ** 2)
    ss_tot = np.sum((log_v - log_v.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return A, alpha, r2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_finetune_best(cfg, sweep_name):
    """
    Returns { dataset: { compute: best_val_loss } } from DyHPO cell sub-sweeps.
    """
    batchsize      = int(cfg.get("fixed_params", {}).get("training.batchsize", 512))
    datasets       = cfg["datasets"]
    t_steps_values = cfg["t_steps_values"]
    best = {}

    for dataset in datasets:
        ds_tag = _ds_tag(dataset)
        for t_steps in t_steps_values:
            cell_name = f"{sweep_name}_{ds_tag}_t{t_steps:05d}"
            _, cell_eos = _sweep_dirs(cfg, cell_name)
            results = load_results_from_dir(os.path.join(cell_eos, "results"))
            if not results:
                continue
            compute  = t_steps * batchsize
            best_val = min(float(r.get("val_loss", float("inf"))) for r in results)
            best.setdefault(dataset, {})[compute] = best_val

    return best


def load_solo_best(solo_dir, dataset, batchsize):
    """
    Read results from all _tNNNNN cell dirs of a solo scaling sweep.
    Searches both inside solo_dir (cells as children) and as siblings of solo_dir.
    Returns { compute: best_val_loss }.
    """
    solo_dir = solo_dir.rstrip("/")
    base     = os.path.basename(solo_dir)
    parent   = os.path.dirname(solo_dir)

    best = {}
    if not os.path.isdir(solo_dir):
        print(f"  [warn] solo dir not found: {solo_dir}", file=sys.stderr)
        return best

    # Check both inside solo_dir and as siblings in parent
    search_dirs = [(solo_dir, base), (parent, base)]
    for search_root, prefix in search_dirs:
        for entry in os.listdir(search_root):
            if not entry.startswith(prefix + "_t"):
                continue
            t_str = entry[len(prefix) + 2:]
            try:
                t_steps = int(t_str)
            except ValueError:
                continue
            results_dir = os.path.join(search_root, entry, "results")
            results = load_results_from_dir(results_dir)
            if not results:
                continue
            compute  = t_steps * batchsize
            best_val = min(
                float(r.get("test_loss", r.get("val_loss", float("inf"))))
                for r in results
            )
            best[compute] = best_val

    return best


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_one(ax, dataset, ft_best, solo_best, batchsize):
    title = dataset.replace("_amplitudes", "")

    # Solo scratch
    if solo_best:
        solo_c = sorted(solo_best)
        solo_v = [solo_best[c] for c in solo_c]
        ax.scatter(solo_c, solo_v, color="steelblue", s=60, zorder=3,
                   label="solo scratch (best per cell)")

    # Fine-tune
    if ft_best:
        ft_c = sorted(ft_best)
        ft_v = [ft_best[c] for c in ft_c]
        ax.scatter(ft_c, ft_v, color="darkorange", marker="*", s=120, zorder=4,
                   label="fine-tune (best per fidelity)")

        if len(ft_c) >= 2:
            A, alpha, r2 = fit_power_law(ft_c, ft_v)
            c_fit = np.logspace(math.log10(ft_c[0]), math.log10(ft_c[-1]), 300)
            v_fit = A * c_fit ** (-alpha)
            ax.plot(c_fit, v_fit, color="tomato", lw=1.8,
                    label=rf"$A \cdot N^{{-\alpha}},\ A={A:.2e},\ \alpha={alpha:.3f},\ R^2={r2:.3f}$")

            # Fixed-alpha fit (alpha=2)
            alpha_fixed = 2.0
            log_A_fixed = np.mean(
                [math.log(v) + alpha_fixed * math.log(c) for c, v in zip(ft_c, ft_v)]
            )
            A_fixed = math.exp(log_A_fixed)
            v_fixed = A_fixed * c_fit ** (-alpha_fixed)
            ax.plot(c_fit, v_fixed, color="tomato", lw=1.2, ls="--",
                    label=rf"fixed-$\alpha$ fit ($\alpha={alpha_fixed}$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("compute (samples seen)")
    ax.set_ylabel("val loss")
    ax.set_title(f"{title} — fine-tune vs solo scaling")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune-config", action="append", required=True,
                        help="Outer sweep_config.yaml of a finetune_scaling sweep "
                             "(repeat for multiple datasets)")
    parser.add_argument("--solo-dir", action="append", required=True,
                        help="Top-level dir of the matching solo scaling sweep "
                             "(e.g. sweeps/scaling_solo_007_eeuunlovirte4). "
                             "Repeat in same order as --finetune-config.")
    parser.add_argument("--out", default=None,
                        help="Output PDF path (default: next to first finetune sweep config)")
    args = parser.parse_args()

    if len(args.finetune_config) != len(args.solo_dir):
        sys.exit("--finetune-config and --solo-dir must be given the same number of times.")

    pairs = list(zip(args.finetune_config, args.solo_dir))
    n_panels = sum(len(yaml.safe_load(open(fc))["datasets"]) for fc, _ in pairs)

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    col = 0

    for ft_config_path, solo_dir in pairs:
        with open(ft_config_path) as f:
            cfg = yaml.safe_load(f)

        sweep_name = cfg["sweep_name"]
        batchsize  = int(cfg.get("fixed_params", {}).get("training.batchsize", 512))
        ft_best    = load_finetune_best(cfg, sweep_name)

        for dataset in cfg["datasets"]:
            solo_best = load_solo_best(solo_dir, dataset, batchsize)
            plot_one(axes[0, col], dataset, ft_best.get(dataset, {}), solo_best, batchsize)
            col += 1

    fig.tight_layout()

    if args.out:
        out_path = args.out
    else:
        with open(pairs[0][0]) as f:
            first_cfg = yaml.safe_load(f)
        top_dir, _ = _sweep_dirs(first_cfg, first_cfg["sweep_name"])
        out_path = os.path.join(os.path.dirname(top_dir),
                                f"{first_cfg['sweep_name']}_vs_solo.pdf")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
