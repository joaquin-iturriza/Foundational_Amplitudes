#!/usr/bin/env python3
"""
fit_scaling_law.py  —  Fit power-law scaling curves from scaling sweep results.

For each dataset, finds the best val_loss across HP trials at each compute
budget, then fits:

    val_loss = A * compute^{-alpha}

Usage (new DyHPO-based scaling sweep):
    python sweep/fit_scaling_law.py --config sweeps/my_scaling_sweep/sweep_config.yaml

Usage (legacy Sobol sweep, explicit dirs):
    python sweep/fit_scaling_law.py --sweep-dir .../sweeps/scaling_solo_001 \\
                                    --eos-dir   .../sweeps/scaling_solo_001
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import yaml

# Reuse the exact FLOPs/step model from the sweep generator so the scaling-law x-axis
# is real training compute (FLOPs), not just samples seen.
_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)
from sweep.generate_pretraining_scaling_sweeps import flops_per_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ds_tag(name):
    return (name.replace('_amplitudes', '').replace(', ', '_').replace('_', ''))


def _n_particles(cfg, ds):
    """Average particles per event = (ncols - 1) / 5 from the npy column count
    (fixed multiplicity per dataset). Feeds the FLOPs/step model."""
    data_path = (cfg.get("fixed_params", {}).get("data.data_path")
                 or os.path.join(cfg["paths"].get("project_dir", ""), "data"))
    path = os.path.join(data_path, ds + ".npy")
    try:
        ncols = np.load(path, mmap_mode="r").shape[1]
        return (ncols - 1) // 5
    except Exception:
        return 5.0   # fall back to the generator's N_AVG


def _sweep_dirs_for(cfg, sweep_name):
    if "sweep_dir" in cfg.get("paths", {}):
        d = os.path.join(cfg["paths"]["sweep_dir"], sweep_name)
        return d, d
    afs = os.path.join(cfg["paths"]["afs_sweep_dir"], sweep_name)
    eos = os.path.join(cfg["paths"]["eos_sweep_dir"], sweep_name)
    return afs, eos


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

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
        except Exception as e:
            print(f"  [warn] {fname}: {e}", file=sys.stderr)
            continue
        results.append(r)
    return results


def collect_best_from_config(cfg, sweep_name):
    """
    New DyHPO structure: one cell per (dataset, t_steps).
    Returns { dataset: { compute_FLOPs: best_val_loss } }, where compute is real
    training FLOPs = flops_per_step(num_heads, n_avg_particles, batchsize) * t_steps
    (so nh=8 cells cost ~4x nh=4 cells at the same t_steps — a fair x-axis).
    """
    batchsize      = int(cfg.get("fixed_params", {}).get("training.batchsize", 512))
    num_heads      = int(cfg.get("fixed_params", {}).get("model.net.num_heads", 8))
    datasets       = cfg["datasets"]
    t_steps_values = cfg["t_steps_values"]
    best = {}

    for dataset in datasets:
        ds_tag = _ds_tag(dataset)
        n_avg  = _n_particles(cfg, dataset)
        fps    = flops_per_step(num_heads, n_avg, batchsize)   # FLOPs per training step
        for t_steps in t_steps_values:
            cell_name = f"{sweep_name}_{ds_tag}_t{t_steps:05d}"
            _, cell_eos = _sweep_dirs_for(cfg, cell_name)
            results_dir = os.path.join(cell_eos, "results")

            results = load_results_from_dir(results_dir)
            if not results:
                print(f"  [warn] No results in {results_dir}")
                continue

            compute  = fps * t_steps
            best_val = min(float(r.get("val_loss", float("inf"))) for r in results)

            best.setdefault(dataset, {})[compute] = best_val
            print(f"  {dataset}  t={t_steps:>5d}  compute={compute:>10.3e} FLOPs  "
                  f"best_val={best_val:.5f}  ({len(results)} trials)")

    return best


def collect_best_from_dir(results_dir):
    """Legacy Sobol structure: single results dir, dataset/compute in JSON."""
    results = load_results_from_dir(results_dir)
    best = {}
    for r in results:
        if "val_loss" not in r or "compute_ds" not in r or "dataset" not in r:
            continue
        ds       = r["dataset"]
        val_loss = float(r["val_loss"])
        compute  = int(r["compute_ds"][ds])
        if ds not in best:
            best[ds] = {}
        if compute not in best[ds] or val_loss < best[ds][compute]:
            best[ds][compute] = val_loss
    return best


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_power_law(compute_vals, val_losses):
    log_c = np.array([math.log(c) for c in compute_vals])
    log_v = np.array([math.log(v) for v in val_losses])
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


def build_solo_reference(anchor_config_path, slope_config_path):
    """Matched-architecture solo reference line for the finetune plots.

    Returns { dataset: (c_anchor_FLOPs, val_anchor, alpha_solo) }: a single nh=8
    SOLO point (from the anchor sweep, real nh=8 FLOPs) plus the per-dataset slope
    alpha taken from a (nh=4) solo scaling sweep. The scaling exponent is assumed
    width-independent, so anchor + slope gives the nh=8 solo line to compare against.
    """
    with open(anchor_config_path) as f:
        a_cfg = yaml.safe_load(f)
    with open(slope_config_path) as f:
        s_cfg = yaml.safe_load(f)
    anchor_best = collect_best_from_config(a_cfg, a_cfg["sweep_name"])   # {ds:{flops:val}} (nh8)
    slope_best  = collect_best_from_config(s_cfg, s_cfg["sweep_name"])   # {ds:{flops:val}} (nh4)

    ref = {}
    for ds, cell in anchor_best.items():
        if not cell:
            continue
        c_a = min(cell)            # single anchor point (one t_steps)
        v_a = cell[c_a]
        sb = slope_best.get(ds, {})
        if len(sb) >= 2:
            cs = sorted(sb); vs = [sb[c] for c in cs]
            _, alpha_solo, _ = fit_power_law(cs, vs)
            ref[ds] = (c_a, v_a, alpha_solo)
        else:
            print(f"  [warn] solo reference: no >=2-point slope for {ds}, skipping")
    return ref


# ---------------------------------------------------------------------------
# alpha vs final-state multiplicity
# ---------------------------------------------------------------------------

# Datasets in the same family are connected (dotted) in order of increasing
# final-state multiplicity. Matched by prefix so the energy / e4 / ratio tails
# don't matter. EDIT to taste.
ALPHA_FAMILIES = {
    "ee→uu(+g)":       ["ee_uu_91", "ee_uug_91", "ee_uugg_91"],
    "ee→aa(+a)":       ["ee_aa_10", "ee_aaa_10"],
    "ee→VV":           ["ee_WW_162", "ee_wwz_255"],
    "ee→ttbar":        ["ee_ttbar_346"],
    # virtual corrections kept as separate points (not connected)
    "ee→uu (virt)":    ["ee_uu_nlo_virt_e4"],
    "ee→ttbar (virt)": ["ee_ttbar_nlo_virt_e4"],
    "ee→uu (virt/born)":    ["ee_uu_nlo_virt_ratio"],
    "ee→ttbar (virt/born)": ["ee_ttbar_nlo_virt_ratio"],
}


def _dofs(n_fs):
    """Independent Lorentz invariants the amplitude depends on (4D, masses fixed):
    for N = n_fs + 2 external legs, 4N - N(on-shell) - 4(mom. cons.) - 6(Lorentz)
    = 3N - 10 = 3*n_fs - 4."""
    return 3 * n_fs - 4


def alpha_lower_bound(n_fs):
    """Theoretical lower bound on the compute exponent: alpha = 4 / DOFs."""
    d = _dofs(n_fs)
    return 4.0 / d if d and d > 0 else None


def _n_final_state(cfg, ds):
    """Final-state particle count = total particles - 2 (incoming e+ e-).
    Total particles inferred from the dataset's column count: ncols = 5*n + 1."""
    data_path = (cfg.get("fixed_params", {}).get("data.data_path")
                 or os.path.join(cfg["paths"].get("project_dir", ""), "data"))
    path = os.path.join(data_path, ds + ".npy")
    try:
        ncols = np.load(path, mmap_mode="r").shape[1]
        return (ncols - 1) // 5 - 2
    except Exception as e:
        print(f"  [warn] final-state count for {ds}: {e}", file=sys.stderr)
        return None


def plot_alpha_vs_multiplicity(best, params_out, cfg, out_dir, skip_ratio=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    info = {}   # ds -> (n_fs, dof, alpha)
    for ds, p in params_out.items():
        if skip_ratio and "ratio" in ds:
            continue
        nfs = _n_final_state(cfg, ds)
        if nfs is not None:
            info[ds] = (nfs, _dofs(nfs), p["alpha"])
    if not info:
        print("  [warn] no final-state counts — skipping alpha-vs-multiplicity plot")
        return

    fam_of = {}
    for fam, members in ALPHA_FAMILIES.items():
        for ds in info:
            if any(ds.startswith(m) for m in members):
                fam_of[ds] = fam

    from matplotlib.ticker import NullFormatter

    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = plt.get_cmap("tab10")
    # x = DOFs (= 3*n_fs - 4): on log-log the bound alpha = 4/DOFs is the straight
    # slope-(-1) diagonal that splits the plot. Points are placed at their DOFs.
    fams = sorted({fam_of.get(ds, "other") for ds in info})
    for i, fam in enumerate(fams):
        members = sorted((ds for ds in info if fam_of.get(ds, "other") == fam),
                         key=lambda d: info[d][1])
        xs = [info[d][1] for d in members]   # DOFs
        ys = [info[d][2] for d in members]   # alpha
        ax.plot(xs, ys, ":", marker="o", ms=6, color=cmap(i % 10), label=fam)

    ax.set_xscale("log")
    ax.set_yscale("log")

    dof_vals = sorted({v[1] for v in info.values()})
    d_min, d_max = dof_vals[0], dof_vals[-1]
    alphas = [v[2] for v in info.values()]
    a_min, a_max = min(alphas), max(alphas)

    # Set the limits FROM the bound so the diagonal alpha = 4/x runs exactly
    # corner-to-corner: top-left = (xlo, 4/xlo), bottom-right = (xhi, 4/xhi).
    # Widen x just enough that every data point (DOFs, alpha) still fits inside.
    m   = 1.15
    xlo = min(d_min, 4.0 / a_max) / m
    xhi = max(d_max, 4.0 / a_min) * m
    yhi = 4.0 / xlo                           # bound at the left edge  -> top-left corner
    ylo = 4.0 / xhi                           # bound at the right edge -> bottom-right corner

    grid = np.logspace(math.log10(xlo), math.log10(xhi), 300)
    bnd  = 4.0 / grid                         # alpha = 4 / DOFs  -> straight diagonal
    ax.plot(grid, bnd, "--", color="gray", lw=1.2)
    ax.fill_between(grid, ylo, bnd, color="gray", alpha=0.15)
    ax.text(0.55, 0.30, "Theoretical lower bound", transform=ax.transAxes,
            rotation=-45, color="gray", fontsize=10, ha="center", va="center")

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Final state particles", fontsize=12)
    ax.set_ylabel(r"$\alpha_C$", fontsize=13)
    # ticks at the DOFs positions, labelled with the final-state particle count
    dof_to_nfs = {v[1]: v[0] for v in info.values()}
    ax.set_xticks(dof_vals)
    ax.set_xticklabels([str(dof_to_nfs[d]) for d in dof_vals])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"alpha_vs_multiplicity.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Plot: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fit power-law scaling from scaling sweep")
    parser.add_argument("--config",    default=None,
                        help="Outer sweep_config.yaml (new DyHPO structure)")
    parser.add_argument("--sweep-dir", default=None,
                        help="Legacy: AFS sweep dir")
    parser.add_argument("--eos-dir",   default=None,
                        help="Legacy: EOS sweep dir (defaults to --sweep-dir)")
    parser.add_argument("--skip-ratio-alpha", action="store_true",
                        help="Exclude the virt/born ratio datasets from the "
                             "alpha-vs-multiplicity plot (other plots unaffected)")
    parser.add_argument("--compare-anchor", default=None,
                        help="Outer config of the nh=8 SOLO anchor sweep (single t_steps). "
                             "Overlays a matched-architecture solo line on each dataset plot.")
    parser.add_argument("--compare-slope", default=None,
                        help="Outer config of the (nh=4) solo scaling sweep to take the "
                             "per-dataset slope alpha from (used with --compare-anchor).")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        sweep_name = cfg["sweep_name"]
        top_dir, _ = _sweep_dirs_for(cfg, sweep_name)
        out_dir    = top_dir

        print(f"Sweep: {sweep_name}")
        best = collect_best_from_config(cfg, sweep_name)

    elif args.sweep_dir:
        sweep_dir   = args.sweep_dir
        eos_dir     = args.eos_dir or sweep_dir
        out_dir     = eos_dir
        results_dir = os.path.join(eos_dir, "results")

        print(f"Reading results from: {results_dir}")
        best = collect_best_from_dir(results_dir)
        print(f"  {sum(len(v) for v in best.values())} cells across {len(best)} datasets")
    else:
        sys.exit("Provide either --config or --sweep-dir.")

    if not best:
        sys.exit("No valid results found.")

    print()
    print(f"{'Dataset':<35}  {'A':>10}  {'alpha':>8}  {'R²':>6}  {'N':>4}")
    print("-" * 68)

    params_out = {}
    for ds in sorted(best):
        cell = best[ds]
        if len(cell) < 2:
            print(f"{ds:<35}  (too few cells: {len(cell)})")
            continue
        computes   = sorted(cell)
        val_losses = [cell[c] for c in computes]
        A, alpha, r2 = fit_power_law(computes, val_losses)
        params_out[ds] = {"A": A, "alpha": alpha, "r2": r2}
        print(f"{ds:<35}  {A:>10.4e}  {alpha:>8.4f}  {r2:>6.3f}  {len(cell):>4}")

    os.makedirs(out_dir, exist_ok=True)
    params_path = os.path.join(out_dir, "scaling_law_params.json")
    with open(params_path, "w") as f:
        json.dump(params_out, f, indent=2)
    print(f"\nScaling law params: {params_path}")

    # matched-architecture solo reference (anchor point + nh4 slope) to overlay
    solo_ref = {}
    if args.compare_anchor and args.compare_slope:
        print("\nBuilding solo reference (anchor + slope) ...")
        solo_ref = build_solo_reference(args.compare_anchor, args.compare_slope)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_ds = len(params_out)
        fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4), squeeze=False)

        for ax, ds in zip(axes[0], sorted(params_out)):
            cell       = best[ds]
            computes   = sorted(cell)
            val_losses = [cell[c] for c in computes]
            p          = params_out[ds]

            ax.scatter(computes, val_losses, color="steelblue", zorder=3, label="best per cell")
            c_fit = np.logspace(math.log10(computes[0]), math.log10(computes[-1]), 200)
            v_fit = p["A"] * c_fit ** (-p["alpha"])
            ax.plot(c_fit, v_fit, color="tomato",
                    label=rf"fit  $\alpha={p['alpha']:.3f}$")
            if ds in solo_ref:
                c_a, v_a, alpha_s = solo_ref[ds]
                v_solo = v_a * (c_fit / c_a) ** (-alpha_s)
                ax.plot(c_fit, v_solo, color="gray", ls="--",
                        label=rf"solo (anchor+slope) $\alpha={alpha_s:.3f}$")
                ax.scatter([c_a], [v_a], color="gray", marker="s", s=40, zorder=4)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("compute  (FLOPs)")
            ax.set_ylabel("val_loss")
            ax.set_title(ds.replace("_amplitudes", ""), fontsize=9)
            ax.legend(fontsize=7)

        fig.suptitle("Scaling laws", fontsize=11)
        fig.tight_layout()
        plot_path = os.path.join(out_dir, "scaling_law.png")
        pdf_path  = os.path.join(out_dir, "scaling_law.pdf")
        fig.savefig(plot_path, dpi=150)
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot: {plot_path}")
        print(f"Plot: {pdf_path}")

        # ── combined overlay: every dataset on one shared log-log axes ─────────
        figc, axc = plt.subplots(figsize=(8, 6))
        cmap = plt.get_cmap("tab20" if len(params_out) > 10 else "tab10")
        for i, ds in enumerate(sorted(params_out)):
            cell       = best[ds]
            computes   = sorted(cell)
            val_losses = [cell[c] for c in computes]
            p          = params_out[ds]
            color      = cmap(i % cmap.N)
            axc.scatter(computes, val_losses, color=color, s=25, zorder=3)
            c_fit = np.logspace(math.log10(computes[0]), math.log10(computes[-1]), 200)
            v_fit = p["A"] * c_fit ** (-p["alpha"])
            axc.plot(c_fit, v_fit, color=color,
                     label=rf"{ds.replace('_amplitudes','')}  ($\alpha$={p['alpha']:.3f})")
            if ds in solo_ref:
                c_a, v_a, alpha_s = solo_ref[ds]
                axc.plot(c_fit, v_a * (c_fit / c_a) ** (-alpha_s),
                         color=color, ls="--", alpha=0.6)
                axc.scatter([c_a], [v_a], color=color, marker="s", s=30, zorder=4)
        axc.set_xscale("log")
        axc.set_yscale("log")
        axc.set_xlabel("compute  (FLOPs)")
        axc.set_ylabel("val_loss")
        axc.set_title("Scaling laws (all datasets)")
        axc.grid(True, which="both", linewidth=0.4, alpha=0.4)
        axc.legend(fontsize=7, ncol=2, frameon=False)
        figc.tight_layout()
        comb_png = os.path.join(out_dir, "scaling_law_combined.png")
        comb_pdf = os.path.join(out_dir, "scaling_law_combined.pdf")
        figc.savefig(comb_png, dpi=150)
        figc.savefig(comb_pdf, bbox_inches="tight")
        plt.close(figc)
        print(f"Plot: {comb_png}")
        print(f"Plot: {comb_pdf}")

    except ImportError:
        print("matplotlib not available — skipping plot.")

    # alpha vs final-state multiplicity (needs cfg to locate the .npy column counts)
    if args.config:
        plot_alpha_vs_multiplicity(best, params_out, cfg, out_dir,
                                   skip_ratio=args.skip_ratio_alpha)


if __name__ == "__main__":
    main()
