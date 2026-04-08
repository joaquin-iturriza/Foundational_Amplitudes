#!/usr/bin/env python3
"""
analyze_sweep.py  —  Inspect a completed (or in-progress) Bayesian HPO sweep.

Usage (from lxplus login node):
    python sweeps/analyze_sweep.py sweeps/my_sweep_001/
    python sweeps/analyze_sweep.py sweeps/my_sweep_001/ --top 20
    python sweeps/analyze_sweep.py sweeps/my_sweep_001/ --no-plots

The sweep directory is created by generate_sweep.py and contains both
optuna_journal.log and a frozen copy of sweep_config.yaml.
"""

import argparse
import os
import sys

import optuna
import yaml
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

optuna.logging.set_verbosity(optuna.logging.WARNING)


def open_study(sweep_dir):
    """Load the Optuna study from a sweep directory (created by generate_sweep.py).

    The sweep directory contains both sweep_config.yaml and optuna_journal.log,
    so no separate config path is needed.
    """
    sweep_dir    = os.path.abspath(sweep_dir)
    journal_path = os.path.join(sweep_dir, "optuna_journal.log")
    config_path  = os.path.join(sweep_dir, "sweep_config.yaml")

    if not os.path.exists(journal_path):
        sys.exit(
            f"Journal not found: {journal_path}\n"
            f"Has generate_sweep.py been run for this sweep yet?"
        )
    if not os.path.exists(config_path):
        sys.exit(
            f"Config not found: {config_path}\n"
            f"This sweep was generated before config auto-saving was added.\n"
            f"Fall back to: python analyze_sweep.py <sweep_dir> --config <path>"
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    sweep_name = cfg["sweep_name"]
    storage    = JournalStorage(JournalFileBackend(journal_path))
    study      = optuna.load_study(study_name=sweep_name, storage=storage)
    return study, cfg


def print_summary(study, top_n):
    import pandas as pd

    trials    = study.trials
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed    = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
    running   = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]
    waiting   = [t for t in trials if t.state == optuna.trial.TrialState.WAITING]

    print(f"\n{'='*60}")
    print(f"  Sweep: {study.study_name}")
    print(f"  Total trials : {len(trials)}")
    print(f"  Completed    : {len(completed)}")
    print(f"  Running      : {len(running)}")
    print(f"  Waiting      : {len(waiting)}")
    print(f"  Failed       : {len(failed)}")
    print(f"{'='*60}")

    if not completed:
        print("No completed trials yet.")
        return

    best = study.best_trial
    print(f"\nBest trial:  #{best.number}  |  val_loss = {best.value:.6f}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k} = {v}")

    df = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    df = df[df["state"] == "COMPLETE"].sort_values("value").head(top_n)
    print(f"\nTop-{min(top_n, len(df))} completed trials (sorted by val_loss):\n")
    print(df.to_string(index=False))
    print()


def save_plots(study, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import numpy as np
    except ImportError as e:
        print(f"Skipping plots — missing dependency: {e}")
        return

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        print("Not enough completed trials for plots (need >= 2).")
        return

    plt.rcParams["text.usetex"] = False
    FONTSIZE = 14
    LABEL_FS = 11
    TICK_FS  = 10
    COLORS   = ["black", "#0343DE", "#A52A2A", "darkorange"]

    trial_nums = [t.number for t in completed]
    values     = [t.value  for t in completed]
    best_idx   = int(np.argmin(values))
    best_val   = values[best_idx]

    with PdfPages(out_path) as pdf:

        # --- 1. Optimization history (log scale) ---
        best_so_far = [min(values[:i + 1]) for i in range(len(values))]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(trial_nums, values, color=COLORS[0], s=20, alpha=0.5, zorder=2,
                   label="trial val_loss")
        ax.plot(trial_nums, best_so_far, color=COLORS[1], linewidth=1.5, zorder=3,
                label="best so far")
        ax.scatter(trial_nums[best_idx], best_val,
                   s=120, marker="*", color=COLORS[3], zorder=4, label="best trial")
        ax.set_yscale("log")
        ax.set_xlabel("Trial number", fontsize=LABEL_FS)
        ax.set_ylabel("val_loss", fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS)
        ax.legend(fontsize=TICK_FS, frameon=False, loc="upper right")
        ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
        ax.set_title("Optimization history", fontsize=FONTSIZE)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- 2. Parameter importance (FANOVA) ---
        if len(completed) >= 4:
            try:
                importances = optuna.importance.get_param_importances(study)
                names  = [k.split(".")[-1] for k in importances]   # strip "training." prefix
                scores = list(importances.values())
                n_bars = len(names)
                fig_h  = max(3, n_bars * 0.6 + 1.5)

                fig, ax = plt.subplots(figsize=(9, fig_h))
                ax.barh(names, scores, color=COLORS[1], alpha=0.85)
                ax.set_xlabel("Importance score (FANOVA)", fontsize=LABEL_FS)
                ax.tick_params(labelsize=TICK_FS)
                ax.grid(True, axis="x", linewidth=0.4, alpha=0.4)
                ax.set_title(
                    "Hyperparameter importance — higher = more influence on val_loss",
                    fontsize=FONTSIZE,
                )
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"  Skipping param importances: {e}")

        # --- 3. Hyperparameter vs val_loss scatter ---
        # One subplot per hyperparameter. y-axis in log scale to match plot 1.
        # Gray dots = all trials; orange star + blue crosshairs = best trial.
        param_names = list(completed[0].params.keys())
        n_params    = len(param_names)

        if n_params > 0:
            ncols = min(n_params, 3)
            nrows = (n_params + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
            axes = np.array(axes).flatten() if n_params > 1 else [axes]

            for ax, pname in zip(axes, param_names):
                param_vals = [t.params[pname] for t in completed]
                best_param = param_vals[best_idx]
                dist       = completed[0].distributions[pname]
                is_cat     = isinstance(dist, optuna.distributions.CategoricalDistribution)

                ax.set_yscale("log")
                ax.axhline(best_val, color=COLORS[1], linestyle="--", alpha=0.5, linewidth=1)

                if is_cat:
                    # Categorical values (e.g. [0, 1e-7, 1e-6, 1e-5]) cannot be
                    # plotted on a numeric axis — they'd all appear at ~0 on a
                    # linear scale and the log scale can't include 0.
                    # Map each choice to an evenly-spaced integer position instead.
                    categories = [str(v) for v in dist.choices]
                    x_pos      = [categories.index(str(v)) for v in param_vals]
                    best_x     = categories.index(str(best_param))

                    ax.scatter(x_pos, values,
                               color="#aaaaaa", s=25, alpha=0.75, zorder=2)
                    ax.scatter(best_x, best_val,
                               s=150, marker="*", color=COLORS[3], zorder=5, label="best")
                    ax.axvline(best_x, color=COLORS[1], linestyle="--", alpha=0.5, linewidth=1)
                    ax.set_xticks(range(len(categories)))
                    ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=TICK_FS)
                    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
                else:
                    ax.scatter(param_vals, values,
                               color="#aaaaaa", s=25, alpha=0.75, zorder=2)
                    ax.scatter(best_param, best_val,
                               s=150, marker="*", color=COLORS[3], zorder=5, label="best")
                    ax.axvline(best_param, color=COLORS[1], linestyle="--", alpha=0.5, linewidth=1)
                    p_arr = np.array(param_vals, dtype=float)
                    if p_arr.min() > 0 and np.log10(p_arr.max() / p_arr.min()) > 2:
                        ax.set_xscale("log")
                    ax.grid(True, which="both", linewidth=0.4, alpha=0.4)

                ax.set_xlabel(pname.split(".")[-1], fontsize=LABEL_FS)
                ax.set_ylabel("val_loss", fontsize=LABEL_FS)
                ax.tick_params(labelsize=TICK_FS)
                ax.legend(fontsize=TICK_FS, frameon=False, loc="upper right")

            for ax in axes[n_params:]:
                ax.set_visible(False)

            fig.suptitle(
                "Hyperparameter vs val_loss  (orange ★ = best trial)",
                fontsize=FONTSIZE, y=1.02,
            )
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Plots saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse an HPO sweep",
        epilog=(
            "Example:\n"
            "  python sweeps/analyze_sweep.py sweeps/lloca_lr_l2_wup_etamin_sweep_001_TEST/\n\n"
            "The sweep directory is created by generate_sweep.py and contains\n"
            "sweep_config.yaml and optuna_journal.log."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sweep_dir",
                        help="Path to the sweep directory (e.g. sweeps/my_sweep_001/)")
    parser.add_argument("--top",      type=int, default=10,
                        help="Number of top trials to print (default: 10)")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving plots to PDF")
    args = parser.parse_args()

    study, cfg = open_study(args.sweep_dir)
    sweep_name = cfg["sweep_name"]

    print_summary(study, top_n=args.top)

    if not args.no_plots:
        plot_dir  = os.path.join(cfg["paths"]["project_dir"], "runs", sweep_name)
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{sweep_name}_analysis.pdf")
        save_plots(study, plot_path)


if __name__ == "__main__":
    main()
