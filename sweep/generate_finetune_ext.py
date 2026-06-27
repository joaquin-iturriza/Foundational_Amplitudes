#!/usr/bin/env python3
"""
generate_finetune_ext.py — extend the finetune-method scaling sweeps with two more
high-compute points, for the two ABSOLUTE-virt datasets only.

Adds t_steps = 20000, 40000 cells for
    ee_uu_nlo_virt_e4, ee_ttbar_nlo_virt_e4         (NOT the _ratio datasets)
to each of the existing method sweeps:
    finetune_scaling_virt_002 (standard), finetune_lora_scaling_virt,
    finetune_ewc_scaling_virt, finetune_resethead_scaling_virt,
    finetune_freeze_scaling_virt.

Same cell machinery as generate_scaling_sweep.py (one DyHPO sweep per (dataset,
t_steps) cell, same fixed_params / search_space / n_trials), so each new point is a
sweep done exactly like the others. The difference: it does NOT bump the sweep name
or clobber the saved outer sweep_config.yaml — it creates the new cells UNDER the
existing names and APPENDS the new t_steps to each outer config's t_steps_values, so
fit_scaling_law / compare_finetune_methods pick them up as part of the same scaling
law (the missing _ratio cells at the new t_steps are simply skipped).

Walltime: t=40000 ≈ 5.7 h at ~0.51 s/iter (well within the 20 h qos_gpu-t3 limit);
cells get time=08:00:00 (train + validation/eval headroom).

Run on Jean Zay:
    python sweep/generate_finetune_ext.py            # create + auto-submit (interleaved)
    python sweep/generate_finetune_ext.py --no-submit
    python sweep/generate_finetune_ext.py --dry-run
"""
import argparse
import copy
import os
import sys

import yaml

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from sweep.generate_scaling_sweep import _make_cell_cfg, _ds_tag, _sweep_dirs

LUSTRE = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
SWEEPS_BASE = os.path.join(LUSTRE, "sweeps")

# Live method sweeps to extend (the saved outer config of each is the source of truth).
LIVE_SWEEPS = [
    "finetune_scaling_virt_002",        # standard
    "finetune_lora_scaling_virt",
    "finetune_ewc_scaling_virt",
    "finetune_resethead_scaling_virt",
    "finetune_freeze_scaling_virt",
]
NEW_DATASETS = ["ee_uu_nlo_virt_e4", "ee_ttbar_nlo_virt_e4"]
NEW_T_STEPS = [20000, 40000]
NEW_TIME = "08:00:00"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweeps", nargs="+", default=LIVE_SWEEPS,
                    help="live sweep names to extend (default: the 5 method sweeps)")
    ap.add_argument("--t-steps", nargs="+", type=int, default=NEW_T_STEPS)
    ap.add_argument("--time", default=NEW_TIME, help="SLURM walltime for the new cells")
    ap.add_argument("--no-submit", action="store_true",
                    help="generate cells + DyHPO state but don't submit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan; write/submit nothing")
    args = ap.parse_args()

    # Heavy (torch/dyhpo) cell-creation primitives — only needed when actually writing.
    if not args.dry_run:
        from sweep.generate_sweep import setup_dirs, init_sampler, write_slurm_script

    new_cell_dirs = []
    for name in args.sweeps:
        outer_path = os.path.join(SWEEPS_BASE, name, "sweep_config.yaml")
        if not os.path.exists(outer_path):
            print(f"[skip] no saved config: {outer_path}")
            continue
        outer = yaml.safe_load(open(outer_path))
        if outer.get("sweep_name") != name:
            print(f"[warn] {name}: saved sweep_name={outer.get('sweep_name')!r} (using it for cell names)")
            name = outer["sweep_name"]
        n_trials = outer["n_trials_per_level"]

        # working copy used to build cells: only the walltime differs from the original
        work = copy.deepcopy(outer)
        work["cluster"] = dict(work["cluster"]); work["cluster"]["time"] = args.time

        print(f"\n== {name} ==  +{args.t_steps} × {NEW_DATASETS}  "
              f"({n_trials} trials/cell, time={args.time})")
        for t in args.t_steps:
            for ds in NEW_DATASETS:
                ds_tag = _ds_tag(ds)
                cell_name = f"{name}_{ds_tag}_t{t:05d}"
                top, _ = _sweep_dirs(work, cell_name)
                if os.path.isdir(top):
                    print(f"   [exists] {cell_name} — skipping")
                    continue
                print(f"   {cell_name}")
                if args.dry_run:
                    continue
                cell_cfg = _make_cell_cfg(work, ds, t, cell_name)
                cell_afs, cell_eos = setup_dirs(cell_cfg, cell_name)
                with open(os.path.join(cell_afs, "sweep_config.yaml"), "w") as f:
                    yaml.dump(cell_cfg, f)
                init_sampler(cell_cfg, cell_afs, cell_eos)
                for i in range(n_trials):
                    write_slurm_script(i, cell_cfg, cell_afs,
                                       os.path.join(cell_afs, "sweep_config.yaml"))
                new_cell_dirs.append(cell_afs)

        # append new t_steps to the outer config so fit/compare include them
        if not args.dry_run:
            merged = sorted(set(outer.get("t_steps_values", [])) | set(args.t_steps))
            if merged != outer.get("t_steps_values"):
                outer["t_steps_values"] = merged
                with open(outer_path, "w") as f:
                    yaml.dump(outer, f)
                print(f"   outer t_steps_values → {merged}")

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return
    if not new_cell_dirs:
        print("\nNo new cells created (all already exist?).")
        return

    print(f"\nCreated {len(new_cell_dirs)} new cells.")
    if args.no_submit:
        print("Submit them interleaved (round-robin across cells) with:")
        print("  python sweep/sweep_manager.py submit " + " ".join(new_cell_dirs))
        return
    from sweep.sweep_manager import submit_sweeps
    print("Auto-submitting (interleaved across cells)...")
    submit_sweeps(new_cell_dirs)


if __name__ == "__main__":
    main()
