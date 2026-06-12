#!/usr/bin/env python3
"""
compare_finetune_methods.py — Overlay the scaling laws of several finetune sweeps.

Reads each sweep's saved outer config (sweeps/<name>/sweep_config.yaml), fits a
power law  val_loss = A * compute^-alpha  per dataset (compute = real FLOPs, same as
fit_scaling_law), and plots ALL methods together — one panel per dataset — so you can
compare standard vs LoRA vs EWC vs reset-head vs freeze finetuning on one axes.
Optionally overlays the matched-architecture solo reference line (nh=8 anchor point +
nh=4 slope), the same one fit_scaling_law draws.

Usage:
    python sweep/compare_finetune_methods.py \\
        --configs sweeps/finetune_scaling_virt_002/sweep_config.yaml \\
                  sweeps/finetune_lora_scaling_virt/sweep_config.yaml \\
                  sweeps/finetune_ewc_scaling_virt/sweep_config.yaml \\
                  sweeps/finetune_resethead_scaling_virt/sweep_config.yaml \\
                  sweeps/finetune_freeze_scaling_virt/sweep_config.yaml \\
        --compare-anchor sweeps/scaling_solo_nh8_anchor_virt_002/sweep_config.yaml \\
        --compare-slope  sweeps/scaling_solo_full/sweep_config.yaml \\
        --out-dir sweeps/finetune_method_comparison
"""
import argparse
import math
import os
import re
import sys

import numpy as np
import yaml

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)
from sweep.fit_scaling_law import (collect_best_from_config, fit_power_law,
                                   build_solo_reference)


def _method_label(sweep_name):
    """standard / lora / ewc / resethead / freeze from the sweep name."""
    n = sweep_name.replace("finetune_", "").replace("scaling_virt", "").strip("_")
    n = re.sub(r"_?\d+$", "", n).strip("_")     # drop trailing _002 etc.
    return n or "standard"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", required=True,
                    help="saved outer sweep_config.yaml of each finetune sweep")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="method labels (default: derived from each sweep_name)")
    ap.add_argument("--compare-anchor", default=None)
    ap.add_argument("--compare-slope", default=None)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    methods = []   # (label, best {ds:{compute:val}}, params {ds:{A,alpha,r2}})
    for i, cpath in enumerate(args.configs):
        with open(cpath) as f:
            cfg = yaml.safe_load(f)
        name  = cfg["sweep_name"]
        label = args.labels[i] if args.labels else _method_label(name)
        print(f"== {label}  ({name}) ==")
        best = collect_best_from_config(cfg, name)
        params = {}
        for ds, cell in best.items():
            if len(cell) >= 2:
                cs = sorted(cell); vs = [cell[c] for c in cs]
                A, alpha, r2 = fit_power_law(cs, vs)
                params[ds] = {"A": A, "alpha": alpha, "r2": r2}
        methods.append((label, best, params))

    solo_ref = {}
    if args.compare_anchor and args.compare_slope:
        print("\nBuilding solo reference (anchor + slope) ...")
        solo_ref = build_solo_reference(args.compare_anchor, args.compare_slope)

    datasets = sorted({ds for _, best, _ in methods for ds in best})
    if not datasets:
        sys.exit("No datasets with results across the given sweeps.")
    os.makedirs(args.out_dir, exist_ok=True)

    # alpha summary table
    print(f"\n{'dataset':<28}" + "".join(f"{lab:>12}" for lab, _, _ in methods))
    for ds in datasets:
        row = f"{ds.replace('_amplitudes',''):<28}"
        for _, _, params in methods:
            a = f"{params[ds]['alpha']:.3f}" if ds in params else "-"
            row += f"{a:>12}"
        print(row)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")

    ncol = min(len(datasets), 2) or 1
    nrow = math.ceil(len(datasets) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.2 * nrow), squeeze=False)
    axflat = axes.flatten()
    for ax in axflat[len(datasets):]:
        ax.axis("off")

    for di, ds in enumerate(datasets):
        ax = axflat[di]
        for mi, (label, best, params) in enumerate(methods):
            if ds not in best:
                continue
            cell = best[ds]; cs = sorted(cell); vs = [cell[c] for c in cs]
            color = cmap(mi % 10)
            ax.scatter(cs, vs, color=color, s=22, zorder=3)
            if ds in params:
                p = params[ds]
                cfit = np.logspace(math.log10(cs[0]), math.log10(cs[-1]), 200)
                ax.plot(cfit, p["A"] * cfit ** (-p["alpha"]), color=color,
                        label=rf"{label} ($\alpha$={p['alpha']:.3f})")
            else:
                ax.plot(cs, vs, color=color, label=label)

        if ds in solo_ref:
            c_a, v_a, alpha_s = solo_ref[ds]
            span = [c for _, best, _ in methods if ds in best for c in best[ds]]
            cfit = np.logspace(math.log10(min(span)), math.log10(max(span)), 200)
            ax.plot(cfit, v_a * (cfit / c_a) ** (-alpha_s), color="gray", ls="--",
                    label=rf"solo ($\alpha$={alpha_s:.3f})")
            ax.scatter([c_a], [v_a], color="gray", marker="s", s=40, zorder=4)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("compute  (FLOPs)"); ax.set_ylabel("val_loss")
        ax.set_title(ds.replace("_amplitudes", ""), fontsize=10)
        ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
        ax.legend(fontsize=7)

    fig.suptitle("Finetuning methods — scaling laws", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(args.out_dir, f"finetune_methods_comparison.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Plot: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
