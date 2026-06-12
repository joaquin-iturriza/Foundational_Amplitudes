#!/usr/bin/env python3
"""
analyze_sweep.py  —  Inspect a completed (or in-progress) DyHPO sweep.

Usage (from lxplus login node):
    python sweeps/analyze_sweep.py sweeps/my_sweep_001/
    python sweeps/analyze_sweep.py sweeps/my_sweep_001/ --top 20
    python sweeps/analyze_sweep.py sweeps/my_sweep_001/ --no-plots

The sweep directory is created by generate_sweep.py on AFS and contains
dyhpo_state.pkl and a frozen copy of sweep_config.yaml.
"""

import argparse
import os
import re
import sys

import yaml

# Make sweep/ importable from the project root
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


def _sweep_dirs(cfg, sweep_name):
    if "sweep_dir" in cfg.get("paths", {}):
        d = os.path.join(cfg["paths"]["sweep_dir"], sweep_name)
        return d, d
    afs = os.path.join(cfg["paths"]["afs_sweep_dir"], sweep_name)
    eos = os.path.join(cfg["paths"]["eos_sweep_dir"], sweep_name)
    return afs, eos


def open_sweep(sweep_dir, force_cpu=False):
    sweep_dir   = os.path.abspath(sweep_dir)
    state_path  = os.path.join(sweep_dir, "dyhpo_state.pkl")
    config_path = os.path.join(sweep_dir, "sweep_config.yaml")

    if not os.path.exists(state_path):
        sys.exit(f"DyHPO state not found: {state_path}\nHas generate_sweep.py been run?")
    if not os.path.exists(config_path):
        sys.exit(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    _, eos_dir      = _sweep_dirs(cfg, cfg["sweep_name"])
    eos_output_path = os.path.join(eos_dir, "dyhpo_surrogate")

    from sweep.dyhpo_sampler import DyHPOSampler
    sampler = DyHPOSampler.load(state_path, eos_output_path, force_cpu=force_cpu)
    return sampler, cfg


def _afs_trial_info(afs_sweep_dir: str, sweep_name: str, hp_idx: int) -> list:
    """
    Return a list of dicts describing the AFS trial(s) that ran (hp_idx, nd_tag).
    Each dict has: trial_idx, job_id, out_file, log_file, gpu, start, end.
    """
    out_dir = os.path.join(afs_sweep_dir, sweep_name, "output")
    log_dir = os.path.join(afs_sweep_dir, sweep_name, "log")
    if not os.path.isdir(out_dir):
        return []

    matches = []
    for fname in sorted(os.listdir(out_dir)):
        if not fname.endswith(".out"):
            continue
        fm = re.match(r'trial_(\d+)\.(\d+)\.0\.out$', fname)
        if not fm:
            continue
        trial_idx = int(fm.group(1))
        job_id    = fm.group(2)

        with open(os.path.join(out_dir, fname)) as f:
            content = f.read()

        hp_m = re.search(r'hp_idx=(\d+)', content)
        if not hp_m or int(hp_m.group(1)) != hp_idx:
            continue
        log_candidates = (
            [f for f in os.listdir(log_dir) if f.startswith(f"trial_{trial_idx:04d}.{job_id}.")]
            if os.path.isdir(log_dir) else []
        )
        log_file = os.path.join(log_dir, log_candidates[0]) if log_candidates else None

        gpu, start, end = "n/a", "n/a", "still running"
        if log_file and os.path.exists(log_file):
            with open(log_file) as f:
                lc = f.read()
            gm = re.search(r'DeviceName\s*=\s*"([^"]+)"', lc)
            gpu = gm.group(1) if gm else "not found"
            sm  = re.search(r'001 \([^)]+\) (\d{2}/\d{2} \d{2}:\d{2}:\d{2})', lc)
            em  = re.search(r'005 \([^)]+\) (\d{2}/\d{2} \d{2}:\d{2}:\d{2})', lc)
            start = sm.group(1) if sm else "n/a"
            end   = em.group(1) if em else "still running"

        matches.append(dict(trial_idx=trial_idx, job_id=job_id,
                            out_file=os.path.join(out_dir, fname), log_file=log_file,
                            gpu=gpu, start=start, end=end))
    return matches


def print_summary(sampler, cfg, top_n):
    results  = sampler.all_results()
    best     = sampler.best_result()
    fidelity = cfg["fidelity_schedule"]
    t_full   = fidelity["t_steps"][-1]

    full_results = [r for r in results if r["t_steps"] == t_full]

    print(f"\n{'='*60}")
    print(f"  Sweep: {cfg['sweep_name']}")
    print(f"  Total evaluations (all fidelity levels): {len(results)}")
    print(f"  Full-fidelity evaluations: {len(full_results)}")
    print(f"  HP candidates in pool: {sampler.n_candidates}")
    print(f"{'='*60}")

    if best:
        params, loss = best
        print(f"\nBest val_loss (any fidelity): {loss:.6f}")
        print("Best params:")
        for k, v in params.items():
            print(f"  {k} = {v}")

    if full_results:
        best_full = full_results[0]
        print(f"\nBest full-fidelity val_loss: {best_full['val_loss']:.6f}")
        print("Full-fidelity best params:")
        for k, v in best_full['params'].items():
            print(f"  {k} = {v}")

    # Show AFS log info for the best run (full-fidelity preferred, any fidelity otherwise)
    best_r = full_results[0] if full_results else (results[0] if results else None)
    if best_r:
        sweep_name    = cfg["sweep_name"]
        afs_dir, _    = _sweep_dirs(cfg, sweep_name)
        hp_idx        = best_r["hp_idx"]
        trials        = _afs_trial_info(os.path.dirname(afs_dir), sweep_name, hp_idx)
        print(f"\nBest run logs  (hp_{hp_idx:04d}):")
        if trials:
            for t in trials:
                print(f"  AFS trial_{t['trial_idx']:04d}  (job {t['job_id']})")
                print(f"    GPU:             {t['gpu']}")
                print(f"    Start / End:     {t['start']}  →  {t['end']}")
                print(f"    HTCondor log:    {t['log_file'] or 'not found'}")
                print(f"    Training stdout: {t['out_file']}")
        else:
            print("  (AFS output directory not accessible from this node)")

    print(f"\nTop-{min(top_n, len(results))} results (all fidelity levels):\n")
    for r in results[:top_n]:
        ps = "  ".join(
            f"{k.split('.')[-1]}={v:.3g}" if isinstance(v, float) else f"{k.split('.')[-1]}={v}"
            for k, v in r['params'].items()
        )
        print(f"  hp_{r['hp_idx']:04d}  val_loss={r['val_loss']:.6f}"
              f"  t_steps={r['t_steps']}  {ps}")
    print()


def save_plots(sampler, cfg, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import numpy as np
    except ImportError as e:
        print(f"Skipping plots — missing dependency: {e}")
        return

    results  = sampler.all_results()
    if len(results) < 2:
        print("Not enough completed evaluations for plots (need >= 2).")
        return

    fidelity     = cfg["fidelity_schedule"]
    t_steps_vals = fidelity["t_steps"]
    n_levels     = len(t_steps_vals)

    plt.rcParams["text.usetex"] = False
    FONTSIZE = 14
    LABEL_FS = 11
    TICK_FS  = 10
    COLORS   = ["black", "#0343DE", "#A52A2A", "darkorange", "#2ca02c"]

    import numpy as np

    with PdfPages(out_path) as pdf:

        # --- 1. Optimization history across all evaluations ---
        # Use chronological order so "best so far" is a proper running minimum.
        chron_results = sampler.all_results_chronological()
        values    = [r["val_loss"] for r in chron_results]
        best_so_far = [min(values[:i+1]) for i in range(len(values))]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(range(len(values)), values, color=COLORS[0], s=15, alpha=0.4, zorder=2,
                   label="val_loss (all fidelity)")
        ax.plot(range(len(values)), best_so_far, color=COLORS[1], linewidth=1.5, zorder=3,
                label="best so far")
        best_idx = int(np.argmin(values))
        ax.scatter(best_idx, values[best_idx], s=120, marker="*", color=COLORS[3], zorder=4,
                   label=f"best ({values[best_idx]:.4f})")
        ax.set_yscale("log")
        ax.set_xlabel("Evaluation index (all fidelity levels)", fontsize=LABEL_FS)
        ax.set_ylabel("val_loss", fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)
        ax.legend(fontsize=TICK_FS, frameon=False)
        ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
        ax.set_title("Optimization history", fontsize=FONTSIZE)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- 2. val_loss vs t_steps  (scaling with training budget) ---
        by_level = {t: [] for t in t_steps_vals}
        for r in results:
            if r["t_steps"] in by_level:
                by_level[r["t_steps"]].append(r["val_loss"])
        if any(len(v) > 0 for v in by_level.values()):
            fig, ax = plt.subplots(figsize=(8, 5))
            positions = list(range(1, n_levels + 1))
            data_bp   = [by_level[t] for t in t_steps_vals]
            ax.boxplot([d if d else [float('nan')] for d in data_bp],
                       positions=positions, patch_artist=True,
                       boxprops=dict(facecolor=COLORS[2], alpha=0.5),
                       medianprops=dict(color="black", linewidth=2))
            ax.set_xticks(positions)
            ax.set_xticklabels([f"{t:,}" for t in t_steps_vals], fontsize=TICK_FS)
            ax.set_yscale("log")
            ax.set_xlabel("Training steps (t_steps)", fontsize=LABEL_FS)
            ax.set_ylabel("val_loss", fontsize=LABEL_FS)
            ax.tick_params(labelsize=TICK_FS)
            ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
            ax.set_title("val_loss distribution vs training steps", fontsize=FONTSIZE)
            for i, t in enumerate(t_steps_vals):
                ax.annotate(f"n={len(by_level[t])}", (i+1, ax.get_ylim()[0]),
                            textcoords="offset points", xytext=(0, 5),
                            ha="center", fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # --- 4a. HP importance — Random Forest feature importances ---
        hp_space    = cfg.get("search_space", [])
        param_names = [e["name"] for e in hp_space]
        n_params    = len(param_names)

        if n_params > 0 and len(results) >= 2:
            try:
                from sklearn.ensemble import RandomForestRegressor

                # Use all results (all fidelity levels) — more data than full-fidelity only.
                # Log-transform log-scale params and val_loss so the RF sees sensible scales.
                log_params = {e["name"] for e in hp_space if e["type"] in ("float_log", "int_log")}
                X_imp, y_imp = [], []
                for r in results:
                    row = []
                    for e in hp_space:
                        v = r["params"].get(e["name"], np.nan)
                        row.append(np.log(v) if e["name"] in log_params and v > 0 else float(v))
                    X_imp.append(row)
                    y_imp.append(np.log(r["val_loss"]) if r["val_loss"] > 0 else float('nan'))

                X_imp = np.array(X_imp)
                y_imp = np.array(y_imp)
                mask  = np.isfinite(X_imp).all(axis=1) & np.isfinite(y_imp)
                if mask.sum() >= 2:
                    rf = RandomForestRegressor(n_estimators=200, random_state=42)
                    rf.fit(X_imp[mask], y_imp[mask])
                    importances = rf.feature_importances_
                    order       = np.argsort(importances)[::-1]
                    labels      = [param_names[i].split(".")[-1] for i in order]

                    fig, ax = plt.subplots(figsize=(max(6, n_params * 1.2), 4))
                    bars = ax.bar(range(n_params), importances[order],
                                  color=COLORS[1], alpha=0.8, edgecolor="black", linewidth=0.5)
                    ax.set_xticks(range(n_params))
                    ax.set_xticklabels(labels, fontsize=LABEL_FS, rotation=20, ha="right")
                    ax.set_ylabel("Feature importance (RF Gini)", fontsize=LABEL_FS)
                    ax.set_ylim(0, min(1.0, importances.max() * 1.3))
                    ax.tick_params(labelsize=TICK_FS)
                    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
                    for bar, imp in zip(bars, importances[order]):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                f"{imp:.2f}", ha="center", va="bottom", fontsize=9)
                    n_used = int(mask.sum())
                    reliability = "" if n_used >= 20 else f"  ⚠ low n={n_used}, interpret with caution"
                    ax.set_title(
                        f"HP importance (Random Forest, all fidelity levels){reliability}",
                        fontsize=FONTSIZE,
                    )
                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
            except ImportError:
                print("Skipping HP importance plot — sklearn not available.")

        # --- 4b. HP vs val_loss scatter (full-fidelity preferred, all levels as fallback) ---
        t_full = t_steps_vals[-1]
        full_r = [r for r in results if r["t_steps"] == t_full]
        scatter_r    = full_r if len(full_r) >= 2 else results
        scatter_label = "full fidelity" if len(full_r) >= 2 else "all fidelity levels"

        if len(scatter_r) >= 2 and n_params > 0:
            full_vals     = [r["val_loss"] for r in scatter_r]
            best_full_val = min(full_vals)
            best_full_idx = full_vals.index(best_full_val)

            ncols = min(n_params, 3)
            nrows = (n_params + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
            axes = np.array(axes).flatten() if n_params > 1 else [axes]

            for ax, pname in zip(axes, param_names):
                pvals     = [r["params"].get(pname) for r in scatter_r]
                best_pval = pvals[best_full_idx]
                entry     = next((e for e in hp_space if e["name"] == pname), {})
                is_log    = entry.get("type", "") in ("float_log", "int_log")

                ax.scatter(pvals, full_vals, color="#aaaaaa", s=25, alpha=0.75, zorder=2)
                ax.scatter(best_pval, best_full_val, s=150, marker="*", color=COLORS[3],
                           zorder=5, label="best")
                ax.axvline(best_pval, color=COLORS[1], linestyle="--", alpha=0.5, linewidth=1)
                ax.axhline(best_full_val, color=COLORS[1], linestyle="--", alpha=0.5, linewidth=1)
                ax.set_yscale("log")
                if is_log:
                    ax.set_xscale("log")
                ax.set_xlabel(pname.split(".")[-1], fontsize=LABEL_FS)
                ax.set_ylabel("val_loss", fontsize=LABEL_FS)
                ax.tick_params(labelsize=TICK_FS)
                ax.legend(fontsize=TICK_FS, frameon=False)
                ax.grid(True, which="both", linewidth=0.4, alpha=0.4)

            for ax in axes[n_params:]:
                ax.set_visible(False)

            fig.suptitle(
                f"HP vs val_loss — {scatter_label} (n={len(scatter_r)} evals)  ★ = best",
                fontsize=FONTSIZE, y=1.02,
            )
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # --- 5. Learning curves: val_loss vs t_steps for top-K configs ---
        hp_with_multi = {}
        for hp_idx, obs_dict in sampler._val_loss_history.items():
            by_t: dict = {}
            for combo, vl in obs_dict.items():
                by_t[combo[0]] = vl
            if len(by_t) > 1:
                hp_with_multi[hp_idx] = [by_t[k] for k in sorted(by_t)]

        if hp_with_multi:
            top_k = min(10, len(hp_with_multi))
            sorted_hps = sorted(
                hp_with_multi.items(),
                key=lambda kv: kv[1][-1],
            )[:top_k]

            fig, ax = plt.subplots(figsize=(8, 5))
            for hp_idx, curve in sorted_hps:
                ax.plot(t_steps_vals[:len(curve)], curve, marker='o', linewidth=1.2,
                        label=f"hp_{hp_idx:04d}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("t_steps", fontsize=LABEL_FS)
            ax.set_ylabel("val_loss", fontsize=LABEL_FS)
            ax.legend(fontsize=7, frameon=False, ncol=2)
            ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
            fig.suptitle(f"Scaling curves — top {top_k} configs by final val_loss",
                         fontsize=FONTSIZE)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # --- 6. Transfer ratios — does joint training beat solo scaling law? ---
        # Requires scaling_law.py to have been run (params_file in sweep config).
        sl_cfg     = cfg.get("scaling_law", {})
        sl_params  = {}
        params_file = sl_cfg.get("params_file")
        if params_file and os.path.exists(params_file):
            import json as _json
            with open(params_file) as f:
                sl_params = _json.load(f)

        tr_records = [r for r in results
                      if r.get("proc_val_losses") and r.get("compute_ds")]
        if sl_params and tr_records:
            alpha_priors = sl_cfg.get("alpha_prior", {})
            dataset_names = list(sl_params.keys())
            labels_short  = [ds.replace('_amplitudes', '') for ds in dataset_names]
            n_datasets    = len(dataset_names)

            fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 5), squeeze=False)
            axes = axes[0]

            for ax_i, (ds_name, ax) in enumerate(zip(dataset_names, axes)):
                p      = sl_params.get(ds_name, {})
                alpha  = p.get("alpha", alpha_priors.get(ds_name, 0.5))
                C      = p.get("C")
                if C is None:
                    ax.set_visible(False)
                    continue

                tr_vals = []
                for r in tr_records:
                    compute = r["compute_ds"].get(ds_name, 0)
                    vl      = r["proc_val_losses"].get(ds_name)
                    if compute > 0 and vl and vl > 0:
                        expected = C * (compute ** (-alpha))
                        tr_vals.append(vl / expected)

                if not tr_vals:
                    ax.set_visible(False)
                    continue

                colors = [COLORS[1] if v < 1 else COLORS[2] for v in tr_vals]
                ax.bar(range(len(tr_vals)), tr_vals, color=colors, alpha=0.75, edgecolor="black", linewidth=0.4)
                ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="solo baseline")
                ax.set_yscale("log")
                ax.set_ylabel("Transfer ratio  (joint / solo expected)", fontsize=LABEL_FS)
                ax.set_xlabel("Evaluation index", fontsize=LABEL_FS)
                ax.set_title(f"{labels_short[ax_i]}\nα={alpha:.2f}  C={C:.2e}", fontsize=FONTSIZE - 1)
                ax.tick_params(labelsize=TICK_FS)
                ax.legend(fontsize=TICK_FS, frameon=False)
                ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)

            fig.suptitle(
                "Transfer ratios — values < 1 mean joint training beats solo scaling law",
                fontsize=FONTSIZE,
            )
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        elif not sl_params:
            print("Skipping transfer ratio plot — run scaling_law.py first to get params_file.")

    print(f"Plots saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse a DyHPO HPO sweep",
        epilog=(
            "Examples:\n"
            "  python sweeps/analyze_sweep.py sweeps/my_sweep_001/\n"
            "  python sweeps/analyze_sweep.py sweeps/my_sweep_001/ --top 20\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sweep_dir", help="Path to the AFS sweep directory")
    parser.add_argument("--top",      type=int, default=10)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--cpu",      action="store_true",
                        help="Force surrogate to load on CPU (use on login nodes with no/full GPU)")
    args = parser.parse_args()

    sampler, cfg = open_sweep(args.sweep_dir, force_cpu=args.cpu)
    sweep_name   = cfg["sweep_name"]

    print_summary(sampler, cfg, top_n=args.top)

    if not args.no_plots:
        plot_dir  = os.path.join(cfg["paths"]["project_dir"], "runs", sweep_name)
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{sweep_name}_analysis.pdf")
        save_plots(sampler, cfg, plot_path)


if __name__ == "__main__":
    main()
