#!/usr/bin/env python3
"""
finetune_compute.py — method-aware training cost for the finetune comparison.

The naive comparison (compare_finetune_methods.py + fit_scaling_law.flops_per_step)
charges EVERY finetune method the SAME full forward+backward FLOPs/step. That erases
the whole point of the cheap methods — LoRA only trains rank-r adapters; `freeze`
skips weight-grads for frozen blocks; EWC pays a one-time Fisher pre-pass. This
module provides:

  (a) method_total_flops(...)     — real per-step FLOPs (× t_steps) per method, and
  (b) method_trainable_params(...) — the trainable-parameter count per method
                                     (∝ optimizer-state bytes), the axis where LoRA
                                     actually wins.

It also parses each DyHPO cell's summary.txt to recover the BEST trial's method HPs
(freeze_blocks, lora.rank, …) — these are SWEPT, so the per-step cost varies per
trial and can't be a per-method constant. Result JSONs don't store the HPs, but
summary.txt does ("Best params" block + a per-trial Top-N table).

FLOP MODEL (per training step), decomposing the standard 3×forward heuristic into
forward F, input-grad (≈F) and weight-grad (≈F):
  d = 16·nh ; f_frames = n·131072 ; f_trans = L·n·(24d² + 2nd) ; fwd = BS·(f_frames+f_trans)
  • standard / resethead / ewc : 3·fwd        (all params trained; full backward)
  • freeze(k of L input-side blocks): fwd + fwd(input, full — framesnet/linear_in below
        stay trainable so backprop can't truncate) + BS·(f_frames + (1−k/L)·f_trans)
        (weight-grad skips the k frozen blocks only)
  • lora : fwd + BS·f_trans (input-grad spans the blocks; framesnet/linear_in are frozen
        with nothing trainable below → backprop truncates there) + ~0 (adapter weight-grad
        ∝ rank, negligible).  ≈ 2/3 of full.
  • ewc additionally: + n_fisher · 3·fwd once (Fisher pre-pass; dominates at low t_steps).
"""

import ast
import gzip
import os
import re

# Architecture constants — must match generate_pretraining_scaling_sweeps.flops_per_step.
D_PER_HEAD = 16
NUM_BLOCKS = 8


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------

def _fwd_components(nh, n_avg, batch_size, n_blocks=NUM_BLOCKS):
    """(framesnet, transformer) forward MACs for one BATCH (not per-event)."""
    d = D_PER_HEAD * nh
    f_frames = n_avg * 131_072
    f_trans = n_blocks * n_avg * (24 * d ** 2 + 2 * n_avg * d)
    return batch_size * f_frames, batch_size * f_trans


def method_step_flops(method, nh, n_avg, batch_size, freeze_k=0, n_blocks=NUM_BLOCKS):
    """Per-step training MACs for a finetune method. `method` ∈
    {standard, resethead, ewc, freeze, lora}. freeze_k = #frozen input-side blocks."""
    F_frames, F_trans = _fwd_components(nh, n_avg, batch_size, n_blocks)
    fwd = F_frames + F_trans
    if method == "lora":
        # forward + input-grad over the blocks only (truncates below block 0);
        # adapter weight-grad ∝ rank → neglected.
        return fwd + F_trans
    if method == "freeze":
        k = max(0, min(int(freeze_k), n_blocks))
        weight_grad = F_frames + (1.0 - k / n_blocks) * F_trans   # skip frozen blocks' weight-grad
        return fwd + fwd + weight_grad                            # fwd + input-grad(full) + weight-grad
    # standard / resethead / ewc → full forward+backward
    return 3.0 * fwd


def method_total_flops(method, nh, n_avg, batch_size, t_steps, hp, n_blocks=NUM_BLOCKS,
                       n_fisher=64):
    """Total training MACs to reach this cell: per-step × t_steps, plus EWC's one-time
    Fisher pre-pass (n_fisher forward+backward passes)."""
    step = method_step_flops(method, nh, n_avg, batch_size,
                             freeze_k=len(hp.get("freeze_blocks", []) or []),
                             n_blocks=n_blocks)
    total = step * t_steps
    if method == "ewc":
        F_frames, F_trans = _fwd_components(nh, n_avg, batch_size, n_blocks)
        total += n_fisher * 3.0 * (F_frames + F_trans)
    return total


# ---------------------------------------------------------------------------
# Trainable parameters (∝ optimizer-state bytes: Adam stores 2 moments/param)
# ---------------------------------------------------------------------------

def load_param_breakdown(pretrained_path):
    """Bucket the pretrained checkpoint's parameter tensors by component, so we can
    count trainable params per method. Returns:
        {"total": int, "blocks": {i: nparams}, "non_block": int,
         "block_attn_linears": {i: [(out,in), ...]}}   # qkv/out linear shapes for LoRA
    Loads torch lazily (cluster-only); the .pt.gz is a gzipped torch save."""
    import torch
    opener = gzip.open if pretrained_path.endswith(".gz") else open
    with opener(pretrained_path, "rb") as f:
        obj = torch.load(f, map_location="cpu", weights_only=False)
    sd = obj.get("model", obj) if isinstance(obj, dict) else obj
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()

    blocks, attn_linears = {}, {}
    total = non_block = 0
    blk_re = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
    for name, t in sd.items():
        try:
            n = int(t.numel())
        except Exception:
            continue
        total += n
        m = blk_re.search(name)
        if not m:
            non_block += n
            continue
        i = int(m.group(1))
        blocks[i] = blocks.get(i, 0) + n
        # qkv/out attention linears (2-D weights) → shapes for LoRA adapter sizing
        if name.endswith(".weight") and t.dim() == 2 and (
                "qkv" in name or "out_linear" in name) and "attention" in name:
            attn_linears.setdefault(i, []).append(tuple(t.shape))   # (out, in)
    return {"total": total, "blocks": blocks, "non_block": non_block,
            "block_attn_linears": attn_linears}


def lora_param_count(breakdown, rank):
    """Σ over wrapped (qkv,out) linears of rank·(in+out)."""
    tot = 0
    for shapes in breakdown["block_attn_linears"].values():
        for (out, inn) in shapes:
            tot += rank * (inn + out)
    return tot


def method_trainable_params(method, hp, breakdown):
    """Trainable-parameter count for a method's best-trial config."""
    total = breakdown["total"]
    if method == "lora":
        return lora_param_count(breakdown, int(hp.get("lora_rank", 8)))
    if method == "freeze":
        frozen = hp.get("freeze_blocks", []) or []
        return total - sum(breakdown["blocks"].get(int(i), 0) for i in frozen)
    return total   # standard / resethead / ewc → all params


# ---------------------------------------------------------------------------
# summary.txt parsing (per-cell best trial + its method HPs)
# ---------------------------------------------------------------------------

def parse_cell_summary(cell_dir):
    """Read <cell_dir>/summary.txt → {"best_val": float, "freeze_blocks": list,
    "lora_rank": int|None, "ewc_lambda": float|None} from the 'Best params' block,
    or None if the file is absent/unparseable."""
    path = os.path.join(cell_dir, "summary.txt")
    if not os.path.exists(path):
        return None
    try:
        txt = open(path).read()
    except Exception:
        return None
    mbv = re.search(r"Best val_loss:\s*([0-9.eE+\-]+)", txt)
    if not mbv:
        return None
    out = {"best_val": float(mbv.group(1)), "freeze_blocks": [],
           "lora_rank": None, "ewc_lambda": None}
    block = re.search(r"Best params:\s*\n(.*?)(?:\n\s*\n|\Z)", txt, re.S)
    if block:
        for line in block.group(1).splitlines():
            m = re.match(r"\s*(\S+)\s*=\s*(.+?)\s*$", line)
            if not m:
                continue
            key, val = m.group(1), m.group(2)
            if key.endswith("freeze_blocks"):
                try:
                    out["freeze_blocks"] = ast.literal_eval(val)
                except Exception:
                    out["freeze_blocks"] = []
            elif key.endswith("lora.rank"):
                try:
                    out["lora_rank"] = int(float(val))
                except Exception:
                    pass
            elif key.endswith("ewc.lambda"):
                try:
                    out["ewc_lambda"] = float(val)
                except Exception:
                    pass
    return out


# ---------------------------------------------------------------------------
# Self-test (no torch / no checkpoint needed): FLOP ratios + a summary parse.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    nh, n, BS = 8, 5.0, 16384
    full = method_step_flops("standard", nh, n, BS)
    print("Per-step FLOPs relative to standard (nh=8, n=5, BS=16384):")
    for m, hp in [("standard", {}), ("resethead", {}), ("ewc", {}),
                  ("freeze", {"freeze_blocks": [0, 1]}),
                  ("freeze", {"freeze_blocks": [0, 1, 2, 3]}),
                  ("freeze", {"freeze_blocks": [0, 1, 2, 3, 4, 5]}),
                  ("lora", {})]:
        k = len(hp.get("freeze_blocks", []))
        r = method_step_flops(m, nh, n, BS, freeze_k=k) / full
        tag = f"{m}" + (f"(k={k})" if m == "freeze" else "")
        print(f"  {tag:14s} {r:.3f}")
    if len(sys.argv) > 1:   # optional: parse a cell's summary.txt
        print("\nparse_cell_summary:", parse_cell_summary(sys.argv[1]))
