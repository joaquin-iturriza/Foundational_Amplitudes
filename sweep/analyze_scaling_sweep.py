#!/usr/bin/env python3
"""
analyze_scaling_sweep.py  —  Inspect a scaling sweep (in-progress or complete).

For each (dataset, t_steps) cell it shows:
  • How many trials completed vs total
  • Best loss found and at which trial
  • Cumulative-best curve  →  tells you if more trials would help
  • Loss distribution across all trials
  • HP correlation with loss  →  tells you if the search space is well-targeted

Usage:
    python sweep/analyze_scaling_sweep.py \\
        --config sweeps/finetune_scaling_eeuunlovirt_001/sweep_config.yaml
        --no-plots                             # text summary only
"""

import argparse
import json
import math
import os
import re
import sys

import numpy as np
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _ds_tag(name):
    return (name.replace("_amplitudes", "")
                .replace(", ", "_")
                .replace("_", ""))


def _sweep_dirs(cfg, sweep_name):
    if "sweep_dir" in cfg.get("paths", {}):
        d = os.path.join(cfg["paths"]["sweep_dir"], sweep_name)
        return d, d
    afs = os.path.join(cfg["paths"]["afs_sweep_dir"], sweep_name)
    eos = os.path.join(cfg["paths"]["eos_sweep_dir"], sweep_name)
    return afs, eos


def load_cell_data(cell_dir, n_trials):
    """
    Returns (rows, candidates) where:
      rows       = [(eval_pos, loss), ...] in evaluation order
      candidates = [hp_dict, ...]          in matching evaluation order, or None
    Reads from dyhpo_state.pkl (preferred) or falls back to results JSON files.
    """
    import pickle
    state_path = os.path.join(cell_dir, "dyhpo_state.pkl")
    if os.path.exists(state_path):
        try:
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            eval_order     = state.get("eval_order", [])
            candidates_raw = state.get("candidates_raw", [])
            observations   = state.get("observations", {})
            rows, cands = [], []
            for eval_pos, (cand_idx, fid) in enumerate(eval_order):
                obs = observations.get(cand_idx, {})
                if fid not in obs:
                    continue
                loss = -obs[fid]   # DyHPO stores negated loss
                rows.append((eval_pos, loss))
                cands.append(candidates_raw[cand_idx] if cand_idx < len(candidates_raw) else {})
            return rows, cands or None
        except Exception as e:
            print(f"  [warn] dyhpo_state.pkl load failed ({e}), falling back to JSON", file=sys.stderr)

    # Fallback: load from results JSON files (order by hp index parsed from filename)
    results_dir = os.path.join(cell_dir, "results")
    if not os.path.isdir(results_dir):
        return [], None
    rows = []
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(results_dir, fname)) as f:
                r = json.load(f)
            loss = float(r.get("test_loss", r.get("val_loss", float("inf"))))
            # parse hp index from filename like hp0015_t316_...json
            hp_idx = int(fname.split("_")[0].replace("hp", "")) if fname.startswith("hp") else -1
            rows.append((hp_idx, loss))
        except Exception:
            continue
    rows.sort()
    return rows, None


def cumulative_best(losses):
    best, out = float("inf"), []
    for v in losses:
        best = min(best, v)
        out.append(best)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--sweep-name",  default=None)
    parser.add_argument("--no-plots",    action="store_true")
    args = parser.parse_args()

    cfg        = load_config(args.config)
    sweep_name = args.sweep_name or cfg["sweep_name"]
    batchsize  = int(cfg.get("fixed_params", {}).get("training.batchsize", 512))

    sweep_dir_base = cfg["paths"].get("sweep_dir",
                                      cfg["paths"].get("afs_sweep_dir", ""))

    # Outer configs have 'datasets'; per-cell configs have dataset in fixed_params.
    if "datasets" in cfg:
        datasets     = cfg["datasets"]
        _parent_name = sweep_name
        t_steps_vals = cfg.get("t_steps_values",
                               cfg.get("fidelity_schedule", {}).get("t_steps", []))
    else:
        datasets     = [cfg["fixed_params"]["data.dataset"].strip("[]").strip()]
        _cell_ds_tag = _ds_tag(datasets[0])
        suffix_pat   = re.compile(rf"_{re.escape(_cell_ds_tag)}_t(\d+)$")
        m            = suffix_pat.search(sweep_name)
        _parent_name = sweep_name[:m.start()] if m else sweep_name
        sibling_pat  = re.compile(
            rf"^{re.escape(_parent_name)}_{re.escape(_cell_ds_tag)}_t(\d+)$")
        t_steps_vals = sorted(
            int(sm.group(1))
            for entry in os.listdir(sweep_dir_base)
            if (sm := sibling_pat.match(entry))
            and os.path.isdir(os.path.join(sweep_dir_base, entry))
        ) or cfg.get("fidelity_schedule", {}).get("t_steps", [])

    n_trials  = (cfg.get("n_trials_per_level")
                 or cfg["dyhpo"].get("n_trials", cfg.get("n_trials", 20)))
    n_startup = cfg["dyhpo"].get("n_startup", 10)

    print(f"Sweep      : {sweep_name}")
    print(f"Trials/cell: {n_trials}  ({n_startup} startup + {n_trials - n_startup} surrogate)")
    print()

    all_data = {}   # (dataset, t_steps) -> {"rows": [...], "candidates": [...]}

    for dataset in datasets:
        ds_tag = _ds_tag(dataset)
        print(f"{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        n_levels = len(t_steps_vals)
        for t_steps in t_steps_vals:
            cell_name               = f"{_parent_name}_{ds_tag}_t{t_steps:05d}"
            afs_cell_dir, eos_cell_dir = _sweep_dirs(cfg, cell_name)
            compute                 = t_steps * batchsize

            rows, candidates = load_cell_data(eos_cell_dir, n_trials)
            all_data[(dataset, t_steps)] = {"rows": rows, "candidates": candidates}

            n_done = len(rows)
            if n_done == 0:
                print(f"  t={t_steps:>5d}  compute={compute:>8d}  [ no results yet ]")
                continue

            losses     = [r[1] for r in rows]
            best_loss  = min(losses)
            best_trial = rows[losses.index(best_loss)][0]
            cum_best   = cumulative_best(losses)
            plateau_at = next((i+1 for i, v in enumerate(cum_best) if v == best_loss), n_done)

            print(f"  t={t_steps:>5d}  compute={compute:>8d}  "
                  f"done={n_done:>3d}/{n_trials}  "
                  f"best={best_loss:.5f} (trial {best_trial:>3d}, found after {plateau_at})")

            if n_done >= 5 and plateau_at <= max(1, n_done // 5):
                print(f"           ↳ best found in first 20% — fewer trials may suffice")

        print()

    if args.no_plots:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    for dataset in datasets:
        ds_tag   = _ds_tag(dataset)
        n_levels = len(t_steps_vals)

        # --- Figure 1: convergence curves ---
        fig, axes = plt.subplots(1, n_levels, figsize=(4 * n_levels, 4), squeeze=False)

        for col, t_steps in enumerate(t_steps_vals):
            rows    = all_data[(dataset, t_steps)]["rows"]
            ax      = axes[0, col]
            compute = t_steps * batchsize

            ax.set_title(f"t={t_steps}  (C={compute:,})", fontsize=8)
            ax.set_xlabel("trial index", fontsize=7)
            ax.set_yscale("log")

            if not rows:
                ax.text(0.5, 0.5, "no results", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                continue

            losses = [r[1] for r in rows]
            cum    = cumulative_best(losses)

            ax.scatter(range(len(losses)), losses, s=15, alpha=0.6,
                       color="steelblue", zorder=3, label="trial loss")
            ax.plot(range(len(cum)), cum, color="tomato", lw=1.5,
                    label="cumulative best")
            ax.axvline(n_startup - 0.5, color="gray", lw=0.8, ls="--",
                       label="startup | surrogate")
            ax.set_ylabel("test_loss", fontsize=7)
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=7)

        fig.suptitle(f"{dataset.replace('_amplitudes','')}  —  convergence per level",
                     fontsize=10)
        fig.tight_layout()
        _, top_eos = _sweep_dirs(cfg, sweep_name)
        out_dir    = top_eos
        os.makedirs(out_dir, exist_ok=True)
        conv_path  = os.path.join(out_dir, f"convergence_{ds_tag}.pdf")
        fig.savefig(conv_path)
        plt.close(fig)
        print(f"Convergence plot: {conv_path}")

        # --- Figure 2: HP sensitivity — rows=levels, cols=HPs ---
        hp_keys    = None
        candidates_by_level = {}
        for t_steps in t_steps_vals:
            cands = all_data[(dataset, t_steps)]["candidates"]
            candidates_by_level[t_steps] = cands
            if cands and hp_keys is None:
                hp_keys = list(cands[0].keys())

        if hp_keys:
            n_hp   = len(hp_keys)
            n_rows = n_levels
            try:
                from sklearn.ensemble import RandomForestRegressor
                _has_rf = True
            except ImportError:
                _has_rf = False

            n_cols = n_hp + (1 if _has_rf else 0)
            fig2, axes2 = plt.subplots(
                n_rows, n_cols,
                figsize=(3.5 * n_cols, 3 * n_rows),
                squeeze=False,
            )

            from matplotlib.colors import LogNorm

            all_losses_flat = [
                l for t in t_steps_vals
                for _, l in all_data[(dataset, t)]["rows"]
            ]
            if all_losses_flat:
                global_norm = LogNorm(vmin=min(all_losses_flat),
                                      vmax=max(all_losses_flat))
            cmap = plt.cm.RdYlGn_r

            for row, t_steps in enumerate(t_steps_vals):
                rows       = all_data[(dataset, t_steps)]["rows"]
                candidates = candidates_by_level[t_steps]
                compute    = t_steps * batchsize

                axes2[row, 0].set_ylabel(f"t={t_steps}\n(C={compute:,})\ntest_loss",
                                         fontsize=7)

                if not rows or not candidates:
                    for col in range(n_cols):
                        axes2[row, col].text(0.5, 0.5, "no results",
                                             ha="center", va="center",
                                             transform=axes2[row, col].transAxes,
                                             fontsize=8)
                    continue

                valid = [(tid, l) for tid, l in rows if tid < len(candidates)]
                if not valid:
                    continue
                tids_v, losses_v = zip(*valid)

                for col, key in enumerate(hp_keys):
                    ax = axes2[row, col]
                    xs = [candidates[tid][key] for tid in tids_v]
                    ax.scatter(xs, losses_v, c=losses_v, cmap=cmap,
                               norm=global_norm, s=20, alpha=0.8)
                    if row == 0:
                        ax.set_title(key.replace("training.", "").replace("fine_tune.", ""),
                                     fontsize=8)
                    if all(v > 0 for v in xs):
                        ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.tick_params(labelsize=6)

                if _has_rf and len(valid) >= 5:
                    ax_rf = axes2[row, n_hp]
                    X = np.array([[candidates[tid][k] for k in hp_keys]
                                  for tid in tids_v])
                    y = np.log(np.array(losses_v))
                    rf = RandomForestRegressor(n_estimators=50, random_state=0)
                    rf.fit(X, y)
                    imps  = rf.feature_importances_
                    order = np.argsort(imps)
                    short_keys = [k.replace("training.", "").replace("fine_tune.", "")
                                  for k in hp_keys]
                    bars = ax_rf.barh(range(n_hp), imps[order], color="steelblue")
                    ax_rf.set_yticks(range(n_hp))
                    ax_rf.set_yticklabels([short_keys[i] for i in order], fontsize=6)
                    ax_rf.set_xlabel("RF importance", fontsize=7)
                    for bar, imp in zip(bars, imps[order]):
                        ax_rf.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                                   f"{imp:.2f}", va="center", fontsize=6)
                    if row == 0:
                        ax_rf.set_title("HP importance", fontsize=8)
                    ax_rf.tick_params(labelsize=6)
                elif _has_rf:
                    axes2[row, n_hp].text(0.5, 0.5, "need ≥5 trials",
                                          ha="center", va="center",
                                          transform=axes2[row, n_hp].transAxes,
                                          fontsize=8)

            sm = ScalarMappable(norm=global_norm, cmap=cmap)
            fig2.colorbar(sm, ax=axes2, label="test_loss (log scale)", shrink=0.6)
            fig2.suptitle(
                f"{dataset.replace('_amplitudes', '')}  —  HP sensitivity per level",
                fontsize=10,
            )
            fig2.tight_layout()
            hp_path = os.path.join(out_dir, f"hp_sensitivity_{ds_tag}.pdf")
            fig2.savefig(hp_path)
            plt.close(fig2)
            print(f"HP sensitivity plot: {hp_path}")

    print(f"\nAll plots saved under {out_dir}/")


if __name__ == "__main__":
    main()
