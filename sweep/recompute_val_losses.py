#!/usr/bin/env python3
"""
recompute_val_losses.py — Recompute no-reg val losses for old result JSONs.

Old runs stored val_loss as arithmetic mean of regularized proc_val_losses.
The correct value is geometric mean of no-reg proc_val_losses_no_reg.
This script loads each saved model, runs per-process val inference, and
rewrites the result JSON with proc_val_losses_no_reg and corrected val_loss.

All runs in a cell share the same data and model architecture, so data is
loaded once per cell and only model weights are swapped per run.

Usage:
    python sweep/recompute_val_losses.py [--dry-run] [--cell CELL_NAME]
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, open_dict

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from experiment import AmplitudeExperiment
from base_experiment import _torch_load
from misc import get_device

LUSTRE_BASE = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
SWEEP_BASE  = os.path.join(LUSTRE_BASE, "sweeps", "pretraining_scaling")


def _compute_proc_no_reg(exp):
    exp.model.eval()
    proc_losses_no_reg = {}
    with torch.no_grad():
        for name, loader in exp.proc_val_loaders.items():
            batch_losses = []
            for data in loader:
                _, loss_no_reg, _ = exp._batch_loss(data)
                batch_losses.append(loss_no_reg)
            proc_losses_no_reg[name] = float(np.mean(batch_losses))
    return proc_losses_no_reg


def _load_weights(exp, run_dir, run_idx):
    for fname in [f"model_run{run_idx}.pt", f"model_run{run_idx}.pt.gz",
                  f"model_run{run_idx}_best.pt", f"model_run{run_idx}_best.pt.gz"]:
        path = os.path.join(run_dir, "models", fname)
        if os.path.exists(path) or os.path.exists(path.replace(".pt.gz", ".pt")):
            exp._load_model_weights(path)
            return
    raise FileNotFoundError(f"No checkpoint for run_idx={run_idx} in {run_dir}/models/")


def process_cell(cell_dir, dry_run):
    results_dir = os.path.join(cell_dir, "results")
    if not os.path.isdir(results_dir):
        return

    ckpt_index_path = os.path.join(cell_dir, "checkpoint_index.json")
    if not os.path.exists(ckpt_index_path):
        print(f"  [skip] no checkpoint_index.json")
        return

    with open(ckpt_index_path) as f:
        ckpt_index = json.load(f)

    json_files = sorted(f for f in os.listdir(results_dir) if f.endswith(".json"))
    if not json_files:
        return

    # Find result JSONs that need fixing and their run dirs
    to_fix = []
    for fname in json_files:
        json_path = os.path.join(results_dir, fname)
        with open(json_path) as f:
            result = json.load(f)
        if "proc_val_losses_no_reg" in result:
            continue
        m = re.match(r"hp(\d+)_t\d+_", fname)
        if not m:
            print(f"  [skip] {fname} — can't parse hp_idx")
            continue
        hp_idx = int(m.group(1))
        ckpt = ckpt_index.get(str(hp_idx))
        if ckpt is None:
            print(f"  [skip] {fname} — hp_idx={hp_idx} not in checkpoint_index")
            continue
        to_fix.append((fname, json_path, result, hp_idx, ckpt))

    if not to_fix:
        print(f"  nothing to fix")
        return

    if dry_run:
        for fname, *_ in to_fix:
            print(f"  [dry-run] would fix {fname}")
        return

    # Set up experiment once using the first run's config (same data + arch for all runs in cell)
    _, _, _, _, first_ckpt = to_fix[0]
    first_run_dir = first_ckpt["run_dir"]
    config_path = os.path.join(first_run_dir, "config.yaml")
    if not os.path.exists(config_path):
        print(f"  [skip cell] no config.yaml in {first_run_dir}")
        return

    cfg = OmegaConf.load(config_path)
    with open_dict(cfg):
        cfg.warm_start_idx = None
        cfg.save = False
        cfg.plot = False
        cfg.ema  = False  # no EMA object needed for inference

    exp = AmplitudeExperiment(cfg)
    exp.warm_start = False
    exp.device = get_device()
    exp.dtype = (
        torch.bfloat16
        if cfg.training.float16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if cfg.training.float16
        else torch.float32
    )
    exp.ema = None

    torch.backends.cuda.enable_flash_sdp(cfg.training.enable_flash_sdp)
    torch.backends.cuda.enable_math_sdp(cfg.training.enable_math_sdp)
    torch.backends.cuda.enable_mem_efficient_sdp(cfg.training.enable_mem_efficient_sdp)

    exp.init_physics()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()
    exp._init_regularization()
    exp.init_model()  # instantiates architecture + MuP shapes; weights overwritten below

    for fname, json_path, result, hp_idx, ckpt in to_fix:
        run_dir = ckpt["run_dir"]
        run_idx = ckpt["run_idx"]
        print(f"  {fname}  hp={hp_idx}  run_idx={run_idx}", end="  ", flush=True)
        try:
            _load_weights(exp, run_dir, run_idx)
            proc_no_reg = _compute_proc_no_reg(exp)
            vals = [v for v in proc_no_reg.values() if v is not None and v > 0]
            val_loss = float(np.exp(np.mean(np.log(vals))))

            old = result["val_loss"]
            result["proc_val_losses_no_reg"] = proc_no_reg
            result["val_loss"] = val_loss
            with open(json_path, "w") as f:
                json.dump(result, f)
            print(f"val_loss {old:.5f} → {val_loss:.5f}")
        except Exception as e:
            print(f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cell", default=None, help="process only this cell")
    args = parser.parse_args()

    if not os.path.isdir(SWEEP_BASE):
        sys.exit(f"SWEEP_BASE not found: {SWEEP_BASE}")

    cell_names = sorted(os.listdir(SWEEP_BASE))
    if args.cell:
        cell_names = [c for c in cell_names if c == args.cell]
        if not cell_names:
            sys.exit(f"Cell not found: {args.cell}")

    for cell_name in cell_names:
        if not cell_name.startswith("scaling_p1"):
            continue
        cell_dir = os.path.join(SWEEP_BASE, cell_name)
        if not os.path.isdir(cell_dir):
            continue
        print(f"\n=== {cell_name} ===")
        process_cell(cell_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
