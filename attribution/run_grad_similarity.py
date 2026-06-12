"""
CLI entry point for GradSim dataset-level attribution.

Usage
-----
python attribution/run_grad_similarity.py \\
    --pretrained_run /path/to/pretrain_run \\
    --target_dataset ee_ttbar_347GeV_amplitudes \\
    --data_path      /path/to/data/ \\
    [--finetuned_run /path/to/finetune_run]  # recommended for full fine-tuning
    [--source_datasets d1 d2 ...]            # default: all datasets from pretrain config
    [--n_batches  100]
    [--batch_size 512]
    [--n_events   50000]                     # truncate each dataset for speed
    [--num_workers 4]
    [--run_idx    0]
    [--output     gradsim_scores.json]

If --finetuned_run is given, g_target is computed at the fine-tuned checkpoint
(asks: "which pre-training datasets align with where fine-tuning took the model?").
Recommended for full fine-tuning.  NOT suitable for LoRA (different parameter
names after injection).

Without --finetuned_run, g_target is computed at the pre-trained checkpoint
(asks: "which datasets were aligned with the target task from the start?").
This is required for LoRA fine-tuning and is otherwise a valid upper bound.
"""

import argparse
import gzip
import io
import json
import os
import sys

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_ATTR_DIR = os.path.dirname(os.path.abspath(__file__))
if _ATTR_DIR not in sys.path:
    sys.path.insert(0, _ATTR_DIR)

from particle_ids import ParticleTokenizer
from grad_similarity import build_loss, load_dataset, run_gradsim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _torch_load(path, **kwargs):
    """Load a PyTorch checkpoint, transparently decompressing .pt.gz if needed."""
    if os.path.exists(path):
        return torch.load(path, **kwargs)
    gz_path = path + ".gz"
    if os.path.exists(gz_path):
        with gzip.open(gz_path, 'rb') as f:
            buf = io.BytesIO(f.read())
        return torch.load(buf, **kwargs)
    raise FileNotFoundError(path)

def _load_config(run_dir: str, run_idx: int):
    """Load the saved Hydra config from a run directory."""
    for name in (f"config_{run_idx}.yaml", "config.yaml"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            return OmegaConf.load(path)
    raise FileNotFoundError(
        f"No config found in {run_dir} (tried config_{run_idx}.yaml, config.yaml)"
    )


def _load_model(run_dir: str, run_idx: int, device: torch.device):
    """Instantiate model from config and load checkpoint weights."""
    cfg = _load_config(run_dir, run_idx)
    model = instantiate(cfg.model)

    # Prefer best checkpoint, fall back to current/bare
    candidates = [
        os.path.join(run_dir, "models", f"model_run{run_idx}_best.pt"),
        os.path.join(run_dir, "models", f"model_run{run_idx}_current.pt"),
        os.path.join(run_dir, "models", f"model_run{run_idx}.pt"),
    ]
    ckpt_path = next((p for p in candidates if os.path.exists(p) or os.path.exists(p + ".gz")), None)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found under {run_dir}/models/ for run_idx={run_idx}.\n"
            f"Expected one of:\n" + "\n".join(f"  {p}[.gz]" for p in candidates)
        )

    state = _torch_load(ckpt_path, map_location=device, weights_only=False)
    # Handle checkpoints saved with DDP (keys prefixed by "module.")
    state_dict = state["model"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)

    print(f"Loaded checkpoint: {ckpt_path}", flush=True)
    return model, cfg


def _load_tokenizer(run_dir: str) -> ParticleTokenizer:
    tok_path = os.path.join(run_dir, "particle_tokenizer.json")
    if os.path.exists(tok_path):
        return ParticleTokenizer.load(tok_path)
    print(f"WARNING: no tokenizer found at {tok_path}, using fresh tokenizer", flush=True)
    return ParticleTokenizer()


def _dtype_from_cfg(cfg) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
    }.get(cfg.training.dtype, torch.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GradSim: dataset-level gradient similarity attribution"
    )
    parser.add_argument(
        "--pretrained_run", required=True,
        help="Pre-training run directory (contains config_*.yaml and models/)"
    )
    parser.add_argument(
        "--finetuned_run", default=None,
        help="Fine-tuning run directory. If given, g_target is computed at θ_ft "
             "(recommended for full fine-tuning; do not use with LoRA)"
    )
    parser.add_argument(
        "--target_dataset", required=True,
        help="Target dataset stem (no .npy extension)"
    )
    parser.add_argument(
        "--data_path", required=True,
        help="Directory containing .npy amplitude files"
    )
    parser.add_argument(
        "--source_datasets", nargs="+", default=None,
        help="Source datasets to score (default: all datasets listed in pretrain config)"
    )
    parser.add_argument(
        "--n_batches", type=int, default=100,
        help="Number of batches to average gradient over per dataset (default: 100)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512,
        help="DataLoader batch size (default: 512)"
    )
    parser.add_argument(
        "--n_events", type=int, default=None,
        help="Truncate each dataset to this many events (default: use all)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader num_workers (default: 4)"
    )
    parser.add_argument(
        "--run_idx", type=int, default=0,
        help="Run index for config/checkpoint filenames (default: 0)"
    )
    parser.add_argument(
        "--output", default="gradsim_scores.json",
        help="Output JSON file path (default: gradsim_scores.json)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # --- pre-trained model ---
    model_pre, cfg_pre = _load_model(args.pretrained_run, args.run_idx, device)
    dtype = _dtype_from_cfg(cfg_pre)
    model_pre.to(dtype=dtype)

    # --- optional fine-tuned model for target gradient ---
    model_ft = None
    if args.finetuned_run is not None:
        model_ft, _ = _load_model(args.finetuned_run, args.run_idx, device)
        model_ft.to(dtype=dtype)
        print("Using fine-tuned model for g_target (θ_ft mode)", flush=True)
    else:
        print("Using pre-trained model for g_target (θ_pre mode)", flush=True)

    # --- loss and tokenizer from pre-training config ---
    cfg        = cfg_pre
    loss_fn    = build_loss(cfg.training.loss)
    tokenizer  = _load_tokenizer(args.pretrained_run)
    amp_trafos = list(cfg.data.get("amp_trafos", [])) or None

    # --- source dataset names ---
    source_names = args.source_datasets or list(cfg.data.dataset)
    print(f"\nTarget  : {args.target_dataset}", flush=True)
    print(f"Sources : {source_names}", flush=True)
    print(f"n_batches={args.n_batches}  batch_size={args.batch_size}"
          f"  n_events={args.n_events}\n", flush=True)

    # --- load all datasets with the pre-training tokenizer ---
    target_ds, target_mm, target_ms = load_dataset(
        args.target_dataset, args.data_path, tokenizer,
        dtype=dtype, n_events=args.n_events, amp_trafos=amp_trafos,
    )
    print(f"Loaded target '{args.target_dataset}': {len(target_ds)} events", flush=True)

    source_datasets = {}
    for name in source_names:
        try:
            ds, mm, ms = load_dataset(
                name, args.data_path, tokenizer,
                dtype=dtype, n_events=args.n_events, amp_trafos=amp_trafos,
            )
            source_datasets[name] = (ds, mm, ms)
            print(f"Loaded source '{name}': {len(ds)} events", flush=True)
        except FileNotFoundError as e:
            print(f"WARNING: skipping '{name}' — {e}", flush=True)

    if not source_datasets:
        print("ERROR: no source datasets could be loaded. Exiting.", flush=True)
        sys.exit(1)

    # --- run attribution ---
    print("\n--- computing gradients ---", flush=True)
    result = run_gradsim(
        model_pre=model_pre,
        model_target=model_ft,   # None → falls back to model_pre
        loss_fn=loss_fn,
        target_dataset=target_ds,
        target_mom_mean=target_mm,
        target_mom_std=target_ms,
        source_datasets=source_datasets,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    # --- write output ---
    output = {
        "method":          "GradSim",
        "pretrained_run":  args.pretrained_run,
        "finetuned_run":   args.finetuned_run,
        "target_gradient": "theta_ft" if args.finetuned_run else "theta_pre",
        "target_dataset":  args.target_dataset,
        "n_batches":      args.n_batches,
        "batch_size":     args.batch_size,
        "n_events":       args.n_events,
        **result,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {args.output}", flush=True)
    print("\n--- Attribution ranking (by cosine similarity) ---", flush=True)
    for rank, name in enumerate(result["attribution_ranking"], 1):
        s = result["scores"][name]
        print(f"  {rank:2d}.  cosine={s['cosine']:+.4f}  {name}", flush=True)


if __name__ == "__main__":
    main()
