#!/usr/bin/env python3
"""
analyze_scaling_cells.py — Run the full DyHPO sweep analysis (HP importance,
optimization history, HP-vs-val_loss scatter) for every Phase 1 pretraining
scaling cell, writing one PDF per cell.

Usage:
    python sweep/analyze_scaling_cells.py                  # all Phase 1 cells
    python sweep/analyze_scaling_cells.py --phase 2        # Phase 2 cells
    python sweep/analyze_scaling_cells.py --cell scaling_p1_D1e4_t1000
    python sweep/analyze_scaling_cells.py --out-dir /tmp/analysis
    python sweep/analyze_scaling_cells.py --cpu            # force CPU (login node)
"""

import argparse
import os
import sys

import yaml

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# Support both Jean-Zay Lustre and the local SSHFS mount
_LUSTRE_BASE = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
_MOUNT_BASE  = "/home/joaquin/mnt/jeanzay/Foundational_Amplitudes"
LUSTRE_BASE  = _LUSTRE_BASE if os.path.isdir(_LUSTRE_BASE) else _MOUNT_BASE
SWEEP_BASE   = os.path.join(LUSTRE_BASE, "sweeps", "pretraining_scaling")


class _StateProxy:
    """
    Minimal sampler-like object loaded directly from the DyHPO pickle state,
    bypassing torch/sklearn.  Reconstructs all_results(), best_result(), etc.
    from the raw state dict (candidates_raw, val_loss_history, eval_order).
    """
    def __init__(self, state_path: str, fidelity_grid: list[int]):
        import pickle
        with open(state_path, "rb") as f:
            state = pickle.load(f)

        self.n_candidates   = state["n_candidates"]
        self._fidelity_grid = fidelity_grid
        self._candidates    = state["candidates_raw"]    # list[dict], indexed by hp_idx
        self._val_loss_history = state["val_loss_history"]  # {hp_idx: {(fi,): vl}}
        self._proc_val_loss_history = state.get("proc_val_loss_history", {})
        self._eval_order    = state.get("eval_order", [])   # [(hp_idx, (fi,)), ...]

    def _record(self, hp_idx: int, combo: tuple) -> dict:
        fi       = combo[0]
        t_steps  = self._fidelity_grid[fi] if fi < len(self._fidelity_grid) else fi
        val_loss = self._val_loss_history[hp_idx][combo]
        params   = self._candidates[hp_idx] if hp_idx < len(self._candidates) else {}
        proc     = self._proc_val_loss_history.get(hp_idx, {}).get(combo)
        return dict(hp_idx=hp_idx, val_loss=val_loss, t_steps=t_steps,
                    params=params, proc_val_losses=proc, compute_ds=None)

    def all_results(self) -> list[dict]:
        records = []
        for hp_idx, obs in self._val_loss_history.items():
            for combo in obs:
                records.append(self._record(hp_idx, combo))
        records.sort(key=lambda r: r["val_loss"])
        return records

    def all_results_chronological(self) -> list[dict]:
        records = []
        for hp_idx, combo in self._eval_order:
            if hp_idx in self._val_loss_history and combo in self._val_loss_history[hp_idx]:
                records.append(self._record(hp_idx, combo))
        return records

    def best_result(self):
        results = self.all_results()
        if not results:
            return None
        r = results[0]
        return r["params"], r["val_loss"]


def open_cell(sweep_dir: str, force_cpu: bool = False):
    """Load DyHPO sampler + config from a cell directory.

    Tries the full DyHPO state first (requires torch/sklearn); falls back
    to reading result JSON files directly if the environment is missing deps.
    """
    config_path   = os.path.join(sweep_dir, "sweep_config.yaml")
    state_path    = os.path.join(sweep_dir, "dyhpo_state.pkl")
    surrogate_dir = os.path.join(sweep_dir, "dyhpo_surrogate")
    results_dir   = os.path.join(sweep_dir, "results")

    if not os.path.exists(config_path):
        return None, None

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    fidelity_grid = cfg.get("fidelity_schedule", {}).get("t_steps", [1])

    # Try full DyHPO load first (needs torch + sklearn)
    if os.path.exists(state_path):
        try:
            from sweep.dyhpo_sampler import DyHPOSampler
            sampler = DyHPOSampler.load(state_path, surrogate_dir, force_cpu=force_cpu)
            return sampler, cfg
        except ImportError:
            pass  # fall through to lightweight load below
        except Exception as e:
            print(f"  [warn] DyHPO state load failed: {e}")

        # Lightweight fallback: read pickle directly, no torch/sklearn
        try:
            sampler = _StateProxy(state_path, fidelity_grid)
            return sampler, cfg
        except Exception as e:
            print(f"  [warn] pickle load failed: {e}")

    return None, None


def list_cells(phase: str) -> list[str]:
    if not os.path.isdir(SWEEP_BASE):
        return []
    # Phase "1" includes both scaling_p1_* and scaling_p1ext_* cells.
    prefixes = (f"scaling_p{phase}_", f"scaling_p{phase}ext_") if phase == "1" else (f"scaling_p{phase}_",)
    return sorted(
        os.path.join(SWEEP_BASE, n)
        for n in os.listdir(SWEEP_BASE)
        if any(n.startswith(p) for p in prefixes)
        and os.path.isdir(os.path.join(SWEEP_BASE, n))
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run analyze_sweep diagnostics for all scaling cells."
    )
    parser.add_argument("--phase", choices=["1", "2"], default="1",
                        help="Which phase cells to analyse (default: 1). "
                             "Phase 1 includes both scaling_p1_* and scaling_p1ext_* cells.")
    parser.add_argument("--cell", default=None,
                        help="Analyse a single cell by name (e.g. scaling_p1_D1e4_t1000)")
    parser.add_argument("--out-dir", default=None,
                        help="Directory for output PDFs "
                             "(default: sweeps/pretraining_scaling/cell_analyses/)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU for surrogate (use on login nodes)")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top results to print per cell (default: 10)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(SWEEP_BASE, "cell_analyses")
    os.makedirs(out_dir, exist_ok=True)

    if args.cell:
        cells = [os.path.join(SWEEP_BASE, args.cell)
                 if not os.path.isabs(args.cell) else args.cell]
    else:
        cells = list_cells(args.phase)

    if not cells:
        sys.exit(f"No cells found in {SWEEP_BASE}")

    # Import save_plots from analyze_sweep (reuse all plotting logic)
    from sweep.analyze_sweep import save_plots

    def _print_summary(sampler, cfg, top_n):
        results = sampler.all_results()
        best    = sampler.best_result()
        print(f"  Total evaluations: {len(results)}"
              f"  |  HP pool: {sampler.n_candidates}")
        if best:
            params, loss = best
            print(f"  Best val_loss: {loss:.6f}")
            for k, v in params.items():
                print(f"    {k} = {v}")
        print(f"  Top-{min(top_n, len(results))}:")
        for r in results[:top_n]:
            ps = "  ".join(
                f"{k.split('.')[-1]}={v:.3g}" if isinstance(v, float) else f"{k.split('.')[-1]}={v}"
                for k, v in r["params"].items()
            )
            print(f"    hp_{r['hp_idx']:04d}  val={r['val_loss']:.6f}  {ps}")

    print(f"Analysing {len(cells)} cell(s) → PDFs in {out_dir}\n")
    completed = 0

    for sweep_dir in cells:
        cell_name = os.path.basename(sweep_dir)
        n_json = sum(
            1 for f in os.listdir(os.path.join(sweep_dir, "results"))
            if f.endswith(".json")
        ) if os.path.isdir(os.path.join(sweep_dir, "results")) else 0

        print(f"── {cell_name}  ({n_json} results)")

        if n_json == 0:
            print("   (no results yet — skip)\n")
            continue

        sampler, cfg = open_cell(sweep_dir, force_cpu=args.cpu)
        if sampler is None:
            print("   (failed to load state — skip)\n")
            continue

        results = sampler.all_results()
        if len(results) < 2:
            print(f"   (only {len(results)} result(s) in DyHPO state — skip)\n")
            continue

        _print_summary(sampler, cfg, top_n=args.top)

        out_path = os.path.join(out_dir, f"{cell_name}_analysis.pdf")
        save_plots(sampler, cfg, out_path)
        completed += 1
        print()

    print(f"\nDone. {completed}/{len(cells)} cells analysed.")
    print(f"PDFs written to: {out_dir}")


if __name__ == "__main__":
    main()
