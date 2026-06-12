#!/usr/bin/env python3
"""
generate_pretraining_scaling_sweeps.py — Generate DyHPO HPO sweeps for pretraining scaling laws.

Phase 1: Fix n_heads=16, vary compute (t_steps) for each dataset size D.
         Fit L(C) to find plateau C*(D).

Phase 2: For each (nh, D), train at C*(D) FLOPs to find optimal model size N*(D).
         t_steps = C*(D) / F_step(nh, n_avg, BS(D))

Each cell is an independent 20-trial DyHPO sweep (8 Sobol seeds, single fidelity).

Usage:
    # Phase 1 (generate all compute-scaling sweeps)
    python sweep/generate_pretraining_scaling_sweeps.py --phase 1 [--dry-run] [--auto-submit]

    # Phase 2 (after Phase 1 analysis produces phase2_compute.json)
    python sweep/generate_pretraining_scaling_sweeps.py --phase 2 \\
        --phase2-compute sweeps/pretraining_scaling/phase2_compute.json \\
        [--dry-run] [--auto-submit]
"""

import argparse
import json
import math
import os
import sys

import yaml

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

DATASETS = [
    "ee_wwz_255-1000GeV_amplitudes",
    "ee_WW_162-1000GeV_amplitudes",
    "ee_ttbar_346-1000GeV_amplitudes",
    "ee_uug_91-1000GeV_amplitudes",
    "ee_uugg_91-1000GeV_amplitudes",
    "ee_aa_10-1000GeV_amplitudes",
    "ee_aaa_10-1000GeV_amplitudes",
    "ee_uu_91-1000GeV_amplitudes",
]
N_DATASETS = len(DATASETS)

# Total dataset sizes (train+val+test across all datasets combined)
DATASET_SIZES = {
    "1e3":   1_000,
    "1e3p5": 3_162,
    "1e4":   10_000,
    "1e4p5": 31_623,
    "1e5":   100_000,
}

def _batch_size(d_total: int) -> int:
    """Largest power of 2 strictly less than d_total."""
    p = 1
    while p * 2 < d_total:
        p *= 2
    return p

def _subsample_per_ds(d_total: int) -> int:
    """Per-dataset subsample cap (data.subsample is applied per dataset)."""
    return max(1, d_total // N_DATASETS)

# Phase 1: t_steps levels per dataset size (logarithmically spaced, ≤1 hr at n_heads=16)
PHASE1_T_STEPS = {
    "1e3":   [316, 1000, 3162, 10000, 31623],
    "1e3p5": [100, 316,  1000, 3162,  10000],
    "1e4":   [32,  100,  316,  1000,  3162],
    "1e4p5": [10,  32,   100,  316,   1000,  3162],
    "1e5":   [10,  32,   100,  316,   1000,  3162],
}

# Phase 2: n_heads values to scan
PHASE2_N_HEADS = [2, 4, 8, 16, 32]

# ---------------------------------------------------------------------------
# FLOPs formula (derived from LLOCAMuPTransformer architecture)
#
# Architecture constants:
#   d = 16 * num_heads  (transformer hidden dim, attn_reps="8x0n+2x1n" → attn_reps.dim=16)
#   L = 8               (num_blocks, fixed)
#   mlp_factor = 4
#   multi_query = False → standard multi-head attention
#
# Per transformer block (forward pass, per event with n particles):
#   QKV proj (d→3d):     6 d² n  MACs
#   Attn scores+values:  2 n² d  MACs
#   Output proj (d→d):   2 d² n  MACs
#   MLP (d→4d→d):       16 d² n  MACs
#   Total:              (24 d² + 2 n d) × n
#
# Framesnet (EquiMLP, hidden=128, 2 layers, fixed regardless of nh):
#   F_framesnet(n) ≈ n × 131_072  MACs  (forward only; ×3 for step)
#
# Total FLOPs per training step (forward + backward ≈ 3× forward):
#   F_step(nh, n_avg, BS) = 3 × BS × [F_framesnet(n_avg) + L × n_avg × (24×d² + 2×n_avg×d)]
#   where d = 16 × nh, L = 8
# ---------------------------------------------------------------------------

NUM_BLOCKS = 8
D_PER_HEAD = 16       # attn_reps.dim = 16 → d = 16 * nh
MLP_FACTOR = 4        # unused directly but implicit in the 16 d² n MLP term

# Average particles per event across the 8 datasets.
# ee_uu/aa/WW/ttbar → n=4; ee_uug/aaa/wwz → n=5; ee_uugg → n=6.
N_AVG = 5.0


def flops_per_step(num_heads: int, n_avg: float, batch_size: int) -> float:
    """Total MACs for one training step (forward + backward)."""
    d = D_PER_HEAD * num_heads
    L = NUM_BLOCKS
    f_transformer = L * n_avg * (24 * d**2 + 2 * n_avg * d)
    f_framesnet = n_avg * 131_072          # forward only, fixed at 128 hidden
    return 3.0 * batch_size * (f_framesnet + f_transformer)


def t_steps_from_compute(c_star_macs: float, num_heads: int, batch_size: int) -> int:
    """Convert a compute budget (MACs) to the corresponding number of training steps."""
    fps = flops_per_step(num_heads, N_AVG, batch_size)
    return max(1, round(c_star_macs / fps))


# ---------------------------------------------------------------------------
# SLURM time estimation
#
# Step times (ms) from iteration_time_output.txt, using n=6 (worst case).
# Indexed as STEP_TIME_MS[nh][bs].  Missing entries → OOM.
# ---------------------------------------------------------------------------

STEP_TIME_MS = {
    4:  {256: 37.78,  512: 37.93,  1024: 41.41,  2048:  61.79, 4096: 116.34, 8192: 222.74, 16384: 445.48},
    8:  {256: 38.71,  512: 42.35,  1024: 61.99,  2048: 115.09, 4096: 218.54, 8192: 426.36, 16384: 852.72},
    16: {256: 43.58,  512: 65.18,  1024: 121.60, 2048: 232.97, 4096: 450.69, 8192: 889.11, 16384: None},   # 16384 OOM on 16GB → gpu_p2l
    32: {256: 74.75,  512: 138.56, 1024: 264.76, 2048: 517.18, 4096: 1015.07, 8192: None,  16384: None},   # 8192 OOM on 16GB, 16384 OOM on 32GB
}
# n_heads=2 is not in the table; extrapolate as slightly faster than n_heads=4.
STEP_TIME_MS[2] = {bs: t * 0.95 if t is not None else None
                   for bs, t in STEP_TIME_MS[4].items()}

# Per-nh cap on batch size based on what fits in available VRAM.
# nh=4,8: BS=16384 fits on 16GB (n=6 worst case: ~6.8 GB, ~12.0 GB).
# nh=16:  BS=16384 needs 32GB (~22 GB); BS=8192 fits 16GB (~11.3 GB).
# nh=32:  BS=16384 OOMs even on 32GB (~35 GB); cap at 8192.
BS_CAP = {2: 16384, 4: 16384, 8: 16384, 16: 16384, 32: 8192}


def slurm_time_str(t_steps: int, num_heads: int, batch_size: int) -> str:
    """Return an HH:MM:SS SLURM time limit.

    Overhead per step (data loading, validation, sampler) is roughly fixed at
    ~1.1–1.2 s regardless of batch size.  For long runs this dominates at small
    BS but is amortised at large BS.  We model it explicitly rather than using a
    flat multiplier so that the estimate stays accurate across all BS values.
    """
    ms_per_step = STEP_TIME_MS.get(num_heads, {}).get(batch_size)
    if ms_per_step is None:
        # OOM on gpu_p2 — submitted to gpu_p2l (32 GB). Step time at BS/2 fits
        # 16 GB and is a good proxy for throughput on 32 GB.
        ms_per_step = STEP_TIME_MS.get(num_heads, {}).get(batch_size // 2, 1200.0) * 2.0
    overhead_ms = 0.14 * batch_size   # data loading overhead, ~0.14ms per sample in batch
    total_seconds = t_steps * (ms_per_step + overhead_ms) / 1000.0
    total_seconds = max(total_seconds, 1800)               # at least 30 min
    total_seconds *= 1.1                                   # 10% safety margin
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    # Round up to nearest 30 min
    minutes = math.ceil((hours * 60 + minutes) / 30) * 30
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:00"


def needs_32gb(num_heads: int, batch_size: int) -> bool:
    """True if this (nh, BS) combination exceeds 16GB VRAM at worst-case n=6.

    Uses the EMA VRAM fit: peak ≈ a·BS·n + b·BS + fixed (from vram_scaling_output.txt).
    Threshold is 15 GB to leave headroom.
    """
    N_MAX = 6
    LIMIT_MB = 15_000
    # (a, b, fixed) from EMA fit in vram_scaling_output.txt
    vram_fit = {
        2:  (0.0501, -0.0463,  22),
        4:  (0.0710, -0.0141,  28),
        8:  (0.1129,  0.0522,  46),
        16: (0.1939,  0.1969, 127),
        32: (0.3584,  0.4825, 426),
        64: (0.6921,  1.0124, 1607),
    }
    a, b, fixed = vram_fit.get(num_heads, (0.5, 0.5, 500))
    peak_mb = a * batch_size * N_MAX + b * batch_size + fixed
    return peak_mb > LIMIT_MB


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

LUSTRE_BASE = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"

BASE_CLUSTER = {
    "scheduler": "slurm",
    "account": "itg@v100",
    "request_gpus": 1,
    "cpus_per_task": 8,
}

BASE_PATHS = {
    "sweep_dir": f"{LUSTRE_BASE}/sweeps/pretraining_scaling",
    "project_dir": LUSTRE_BASE,
    "setup_commands": [
        "module load anaconda-py3/2023.09",
        "conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational",
    ],
}

DYHPO = {
    "n_candidates": 300,
    "seed": 42,
    "n_startup": 8,
    "total_budget": 10000,
}

RANGE_EXTENSION = {
    "enabled": False,
    "params": ["training.lr", "training.regularization_lambda"],
    "top_k": 5,
    "boundary_frac": 0.2,
    "trigger_count": 3,
    "extend_factor": 3.0,
    "n_new_candidates": 50,
    "min_observations": 20,
    "max_extensions": 3,
}

LR_LOW_DEFAULT  = 10 ** -4.5   # ≈ 3.16e-5
LR_HIGH_DEFAULT = 0.1

SEARCH_SPACE = [
    {"name": "training.lr",                   "type": "float_log",     "low": LR_LOW_DEFAULT, "high": LR_HIGH_DEFAULT},
    {"name": "training.regularization_lambda","type": "float_log",     "low": 1e-11, "high": 1e-6},
    {"name": "training.cosanneal_warmup_frac","type": "float_uniform", "low": 0.0,   "high": 0.2},
    {"name": "training.cosanneal_eta_min",    "type": "float_log",     "low": 1e-11, "high": 1e-6},
    {"name": "training.ema_decay",            "type": "float_uniform", "low": 0.9,   "high": 0.9999},
    {"name": "training.sampler_alpha_ema",    "type": "float_uniform", "low": 0.3,   "high": 0.95},
    {"name": "training.sampler_min_alpha_frac","type": "float_uniform","low": 0.05,  "high": 0.5},
]

DATASET_LIST_STR = "[" + ", ".join(DATASETS) + "]"
AMP_ORDERS_STR = "[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]"


# ---------------------------------------------------------------------------
# Phase 1 ext: larger model (n_heads=32) with LR centered on Phase 1 optimum
# ---------------------------------------------------------------------------

PHASE1EXT_N_HEADS_LIST = [8, 32]   # default widths; override with --ext-heads

# Same t_steps grid as Phase 1 — at larger nh each step costs more FLOPs,
# naturally extending the compute range with the same grid.
PHASE1EXT_T_STEPS = PHASE1_T_STEPS  # reuse same dict

# D=1e3 already shows a clear plateau at n_heads=16; skip it for the extension.
PHASE1EXT_D_KEYS = ["1e3p5", "1e4", "1e4p5", "1e5"]


def _load_best_lr(d_key: str, t_steps: int) -> float | None:
    """Return the best LR found in the matching Phase 1 n_heads=16 cell, or None."""
    import pickle
    sweep_base = os.path.join(LUSTRE_BASE, "sweeps", "pretraining_scaling")
    candidates = []
    if os.path.isdir(sweep_base):
        for base in (f"scaling_p1_nh16_D{d_key}_t{t_steps}",
                     f"scaling_p1_D{d_key}_t{t_steps}"):
            for name in os.listdir(sweep_base):
                if name == base or name.startswith(base + "_"):
                    candidates.append(os.path.join(sweep_base, name))
    if not candidates:
        return None
    cell_dir = sorted(candidates)[-1]
    state_path = os.path.join(cell_dir, "dyhpo_state.pkl")
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        vh = state.get("val_loss_history", {})
        if not vh:
            return None
        best_hp = min(vh, key=lambda i: min(vh[i].values()))
        cands = state.get("candidates_raw", [])
        if best_hp >= len(cands):
            return None
        return float(cands[best_hp].get("training.lr", 0)) or None
    except Exception:
        return None


def _lr_range_centered(best_lr: float, half_decades: float = 1.0) -> tuple[float, float]:
    """LR search range ±half_decades around best_lr, capped at global defaults."""
    lo = max(LR_LOW_DEFAULT,  best_lr * 10 ** (-half_decades))
    hi = min(LR_HIGH_DEFAULT, best_lr * 10 ** (+half_decades))
    return lo, hi


def make_cell_config(
    cell_name: str,
    num_heads: int,
    t_steps: int,
    d_key: str,
    auto_submit: bool = False,
    lr_range: tuple[float, float] | None = None,
) -> dict:
    d_total = DATASET_SIZES[d_key]
    bs = min(_batch_size(d_total), BS_CAP.get(num_heads, 8192))
    sub = _subsample_per_ds(d_total)
    use_32gb = needs_32gb(num_heads, bs)
    partition = "gpu_p2l" if use_32gb else "gpu_p2"
    slurm_time = slurm_time_str(t_steps, num_heads, bs)

    cluster = {**BASE_CLUSTER, "partition": partition, "time": slurm_time, "auto_submit": auto_submit}

    fixed_params = {
        "data.data_path": f"{LUSTRE_BASE}/data/",
        "data.dataset": DATASET_LIST_STR,
        "data.amp_orders": AMP_ORDERS_STR,
        "data.subsample": sub,
        "data.train_test_val": "[0.7, 0.2, 0.1]",
        "model": "lloca",
        "model.net.num_blocks": 8,
        "model.net.num_heads": num_heads,
        "training.batchsize": bs,
        "evaluation.batchsize": 8192,
        "training.get_ID": "false",
        "training.regularization": "L2",
        "training.save_intermediate": "false",
        "training.scheduler": "CosineAnnealingLR",
        "training.loss_aggregation": "geometric_mean",
        "training.validate_frac": 0.01,
        "seed": 42,                 # fix init seed so trials differ only by hyperparameters
        "plot": "true",
    }

    # Build search space, optionally overriding the LR range
    search_space = list(SEARCH_SPACE)
    if lr_range is not None:
        lr_lo, lr_hi = lr_range
        search_space = [
            {**e, "low": lr_lo, "high": lr_hi} if e["name"] == "training.lr" else e
            for e in search_space
        ]

    return {
        "cluster": cluster,
        "paths": BASE_PATHS,
        "sweep_name": cell_name,
        "n_trials": 20,
        "dyhpo": DYHPO,
        "range_extension": RANGE_EXTENSION,
        "fidelity_schedule": {"t_steps": [t_steps]},
        "fixed_params": fixed_params,
        "search_space": search_space,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate pretraining scaling law sweeps (Phase 1 and/or Phase 2)."
    )
    parser.add_argument("--phase", choices=["1", "1ext", "2", "both", "all"], default="both",
                        help="Which phase to generate (default: both). "
                             "'1ext' = Phase 1 extension with n_heads=32. "
                             "'all' = Phase 1 + 1ext + Phase 2.")
    parser.add_argument("--phase2-compute", metavar="PATH",
                        help="JSON file with C* per dataset size (MACs), "
                             "produced by analyze_pretraining_scaling.py after Phase 1. "
                             "Required when --phase is '2' or 'both'.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without writing files or initialising DyHPO state")
    parser.add_argument("--auto-submit", action="store_true",
                        help="Automatically submit jobs after generating")
    parser.add_argument("--ext-heads", default=None, metavar="NH_LIST",
                        help=f"Comma-separated num_heads for Phase 1 ext "
                             f"(default: {PHASE1EXT_N_HEADS_LIST}). "
                             f"Example: --ext-heads 8,32")
    args = parser.parse_args()

    do_phase1    = args.phase in ("1", "both", "all")
    do_phase1ext = args.phase in ("1ext", "all")
    do_phase2    = args.phase in ("2", "both", "all")

    ext_heads = (
        [int(x.strip()) for x in args.ext_heads.split(",")]
        if args.ext_heads else PHASE1EXT_N_HEADS_LIST
    )

    # Load phase2 compute if needed
    phase2_compute: dict[str, float] = {}
    if do_phase2:
        if args.phase2_compute is None:
            parser.error("--phase2-compute is required when generating Phase 2 sweeps.")
        with open(args.phase2_compute) as f:
            phase2_compute = json.load(f)
        missing = [k for k in DATASET_SIZES if k not in phase2_compute]
        if missing:
            parser.error(f"phase2_compute JSON is missing keys: {missing}")

    # Import generate_sweep helpers
    sys.path.insert(0, os.path.join(_project_dir, "sweep"))
    from sweep.generate_sweep import (setup_dirs, init_sampler, write_slurm_script,
                                       run_generate)

    configs_dir = os.path.join(LUSTRE_BASE, "sweeps", "pretraining_scaling", "configs")
    if not args.dry_run:
        os.makedirs(configs_dir, exist_ok=True)

    cells: list[tuple[str, dict]] = []   # (cell_name, config_dict)

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    if do_phase1:
        print("=== Phase 1: compute scaling (n_heads=16) ===")
        for d_key, d_total in DATASET_SIZES.items():
            bs = _batch_size(d_total)
            for t in PHASE1_T_STEPS[d_key]:
                cell_name = f"scaling_p1_nh16_D{d_key}_t{t}"
                cfg = make_cell_config(cell_name, num_heads=16, t_steps=t,
                                       d_key=d_key, auto_submit=args.auto_submit)
                c_macs = flops_per_step(16, N_AVG, bs) * t
                print(f"  {cell_name:40s}  BS={bs:5d}  t={t:6d}  "
                      f"C={c_macs/1e12:.3f} TMACs  time={cfg['cluster']['time']}")
                cells.append((cell_name, cfg))

    # ── Phase 1 ext (multiple widths) ────────────────────────────────────────
    if do_phase1ext:
        print(f"\n=== Phase 1 ext: compute scaling (n_heads={ext_heads}) ===")
        for nh in ext_heads:
            for d_key in PHASE1EXT_D_KEYS:
                d_total = DATASET_SIZES[d_key]
                bs      = _batch_size(d_total)
                use32gb = needs_32gb(nh, bs)
                for t in PHASE1EXT_T_STEPS[d_key]:
                    best_lr = _load_best_lr(d_key, t)
                    if best_lr is not None:
                        lr_range = _lr_range_centered(best_lr)
                        lr_note  = (f"LR=[{lr_range[0]:.1e},{lr_range[1]:.1e}] "
                                    f"(centered on {best_lr:.1e})")
                    else:
                        lr_range = None
                        lr_note  = "LR=default (no Phase 1 result found)"
                    cell_name = f"scaling_p1ext_nh{nh}_D{d_key}_t{t}"
                    cfg = make_cell_config(cell_name, num_heads=nh, t_steps=t,
                                           d_key=d_key, auto_submit=args.auto_submit,
                                           lr_range=lr_range)
                    c_macs = flops_per_step(nh, N_AVG, bs) * t
                    print(f"  {cell_name:48s}  BS={bs:5d}  t={t:6d}  "
                          f"C={c_macs/1e12:.3f} TMACs  time={cfg['cluster']['time']}"
                          + ("  [gpu_p2l]" if use32gb else "")
                          + f"\n    {lr_note}")
                    cells.append((cell_name, cfg))

    # ── Phase 2 ─────────────────────────────────────────────────────────────
    if do_phase2:
        print("\n=== Phase 2: model size scaling ===")
        for d_key, d_total in DATASET_SIZES.items():
            bs = _batch_size(d_total)
            c_star = phase2_compute[d_key]
            for nh in PHASE2_N_HEADS:
                t = t_steps_from_compute(c_star, nh, bs)
                cell_name = f"scaling_p2_D{d_key}_nh{nh}"
                cfg = make_cell_config(cell_name, num_heads=nh, t_steps=t,
                                       d_key=d_key, auto_submit=args.auto_submit)
                use32 = needs_32gb(nh, bs)
                print(f"  {cell_name:40s}  BS={bs:5d}  t={t:6d}  "
                      f"C={c_star/1e12:.3f} TMACs  time={cfg['cluster']['time']}"
                      + ("  [gpu_p2l]" if use32 else ""))
                cells.append((cell_name, cfg))

    if args.dry_run:
        print(f"\n[dry-run] {len(cells)} cells — no files written.")
        return

    # ── Write configs and initialise sweeps ──────────────────────────────────
    print(f"\nGenerating {len(cells)} sweep cells...")
    cell_dirs: list[str] = []

    for cell_name, cfg in cells:
        config_path = os.path.join(configs_dir, f"{cell_name}.yaml")
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        # Generate the cell (dir setup + DyHPO init) but defer submission, so all
        # cells can be submitted together and interleaved by the sweep_manager.
        try:
            cell_dir = run_generate(config_path, submit=False)
            cell_dirs.append(cell_dir)
        except Exception as e:
            print(f"  ERROR initialising {cell_name}: {e}", file=sys.stderr)
            continue

    print("\nDone.")
    # Each cell is its own sweep (own DyHPO state). The sweep_manager interleaves
    # trials round-robin across all cells so they don't all start in parallel.
    if args.auto_submit:
        if cell_dirs:
            from sweep.sweep_manager import submit_sweeps
            print(f"\nSubmitting {len(cell_dirs)} cells interleaved via sweep_manager...")
            submit_sweeps(cell_dirs)
    elif cell_dirs:
        print("\nTo submit all generated cells interleaved (round-robin across cells):")
        print("  python sweep/sweep_manager.py submit " + " ".join(cell_dirs))


if __name__ == "__main__":
    main()
