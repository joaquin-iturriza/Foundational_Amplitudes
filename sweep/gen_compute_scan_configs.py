"""Emit one generate_scaling_sweep config per (family, datapoint-size D).

Fidelity grid is in COMPUTE (FLOP), not steps: for each D the batch size is
batchsize(D) = min(16384, 0.35*D) (= experiment.py min(cfg, n_train/2)), so the
per-step FLOP differs by D and the step count is solved per cell to hit a shared
compute grid. Cells whose estimated walltime exceeds ~70 min are dropped (the
"~1 hour max runtime" cap) -- this trims the top of the small-D (1k) grids.

Families:
  solo  : from-scratch, search training.lr in [3.16e-5, 0.3]   (nh8)
  ft8   : finetune <- pretrain_full_nh8, search lr_scale [0.005, 50]  (base lr 1.08e-3)
  ft25  : finetune <- pretrain25,       search lr_scale [0.005, 10]  (base lr 8.32e-3)

solo reuses existing 10k/100k sweeps; here it only emits the NEW sizes (1k, 1M) and
a single ~1h extension point for 10k. ft8/ft25 are emitted for all of 1k/10k/100k/1M.
"""
import os, yaml

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compute_scan")
os.makedirs(OUT, exist_ok=True)
ROOT = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"

def f_step(bs):                       # MACs/step for nh8 n_avg4
    return 3.0 * bs * 13139968
def perstep_flop(bs):                 # 2x: MACs -> FLOP
    return 2 * f_step(bs)

def batchsize(D):                     # experiment.py:875  min(cfg_bs, n_train/2)
    return int(min(16384, 0.7 * D / 2))

S_PER_STEP = {350: 0.05, 3500: 0.122, 16384: 0.55}   # measured/estimated wallclock
WALL_CAP_S = 4200                                    # ~70 min

C_GRID = [1e13, 3e13, 1e14, 3e14, 1e15, 3e15, 8e15]
C_GRID_1K = [1e13, 3e13, 1e14, 3e14, 1e15, 2e15]   # small batch is FLOP-inefficient; ~1h tops at 2e15

DATASETS = ["ee_uu_nlo_virt_e4", "ee_ttbar_nlo_virt_e4"]

CLUSTER = {"scheduler": "slurm", "auto_submit": False, "partition": "gpu_p2",
           "account": "itg@v100", "request_gpus": 1, "cpus_per_task": 8, "time": "02:00:00"}
PATHS = {"sweep_dir": f"{ROOT}/sweeps", "project_dir": ROOT,
         "setup_commands": ["module load anaconda-py3/2023.09",
                            "conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational",
                            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]}

BASE_FIXED = {
    "data.data_path": f"{ROOT}/data/",
    "data.train_test_val": "[0.7, 0.2, 0.1]",
    "model": "lloca", "model.net.num_blocks": 8, "model.net.num_heads": 8,
    "training.batchsize": 16384, "evaluation.batchsize": 8192,
    "training.es_load_best_model": "false", "training.get_ID": "false",
    "training.regularization": "L2", "training.save_intermediate": "false",
    "training.scheduler": "CosineAnnealingLR", "training.loss_aggregation": "geometric_mean",
    "training.validate_frac": 0.01, "plot": "true", "seed": 42,
}

SS_COMMON = [
    {"name": "training.regularization_lambda", "type": "float_log", "low": 1.0e-11, "high": 1.0e-06},
    {"name": "training.cosanneal_warmup_frac", "type": "float_uniform", "low": 0.0, "high": 0.2},
    {"name": "training.cosanneal_eta_min", "type": "float_log", "low": 1.0e-11, "high": 1.0e-06},
]

def solo_family():
    fixed = dict(BASE_FIXED)
    ss = [{"name": "training.lr", "type": "float_log", "low": 3.1622776601683795e-05, "high": 0.3}] \
         + SS_COMMON + [{"name": "training.ema_decay", "type": "float_uniform", "low": 0.9, "high": 0.9999}]
    return fixed, ss

def ft_family(pretrained, base_lr, lr_scale_high):
    fixed = dict(BASE_FIXED)
    fixed["fine_tune.pretrained_path"] = pretrained
    fixed["training.lr"] = base_lr
    ss = [{"name": "fine_tune.lr_scale", "type": "float_log", "low": 0.005, "high": lr_scale_high},
          {"name": "fine_tune.layer_decay", "type": "float_uniform", "low": 0.65, "high": 1.0}] + SS_COMMON
    return fixed, ss

FAMILIES = {
    "solo": dict(builder=solo_family(), n_trials=10),
    "ft8":  dict(builder=ft_family(f"{ROOT}/runs/pretrain_full_nh8/trial_0271/models/model_run0_best.pt.gz",
                                   0.001081750557879577, 50.0), n_trials=10),
    "ft25": dict(builder=ft_family(f"{ROOT}/runs/pretrain25/trial_0009/models/model_run0_best.pt.gz",
                                   0.008322839, 10.0), n_trials=10),
}

# which (family -> list of (Dtag, D, restrict_C)) to emit
def t_levels(D, c_grid):
    bs = batchsize(D); ps = perstep_flop(bs); sp = S_PER_STEP[bs]
    out = []
    for C in c_grid:
        t = max(1, round(C / ps))
        wall = t * sp
        if wall <= WALL_CAP_S:
            out.append((C, t, wall))
    return bs, out

PLAN = {
    "solo": [("1k", 1000, C_GRID_1K), ("1M", 1000000, C_GRID),
             ("10kext", 10000, [8e15])],          # only the missing ~1h point for 10k
    "ft8":  [("1k", 1000, C_GRID_1K), ("10k", 10000, C_GRID),
             ("100k", 100000, C_GRID), ("1M", 1000000, C_GRID)],
    "ft25": [("1k", 1000, C_GRID_1K), ("10k", 10000, C_GRID),
             ("100k", 100000, C_GRID), ("1M", 1000000, C_GRID)],
}

total_cells = total_jobs = 0
print(f"{'config':22s} {'D':>8} {'bs':>6} {'levels(t_steps)':<48} {'top_wall':>8}")
for fam, sizes in PLAN.items():
    fixed_base, ss = FAMILIES[fam]["builder"]
    n_tr = FAMILIES[fam]["n_trials"]
    for dtag, D, cg in sizes:
        bs, levels = t_levels(D, cg)
        if not levels: continue
        fixed = dict(fixed_base); fixed["data.subsample"] = D
        cfg = {"cluster": CLUSTER, "paths": PATHS, "sweep_name": f"cscan_{fam}_D{dtag}",
               "seed": 42, "datasets": DATASETS,
               "t_steps_values": [t for _, t, _ in levels], "n_trials_per_level": n_tr,
               "dyhpo": {"n_candidates": 200, "seed": 42, "n_startup": n_tr},  # pure Sobol (concurrent => no BO)
               "fixed_params": fixed, "search_space": ss}
        path = os.path.join(OUT, f"cscan_{fam}_D{dtag}.yaml")
        with open(path, "w") as f: yaml.dump(cfg, f, sort_keys=False)
        ncells = len(levels) * len(DATASETS); njobs = ncells * n_tr
        total_cells += ncells; total_jobs += njobs
        tdesc = " ".join(f"{t}" for _, t, _ in levels)
        print(f"{'cscan_'+fam+'_D'+dtag:22s} {D:>8} {bs:>6} {tdesc:<48} {levels[-1][2]/60:>6.0f}m  "
              f"-> {ncells} cells x{n_tr} = {njobs} jobs")
print(f"\nTOTAL: {total_cells} cells, {total_jobs} jobs across {len(os.listdir(OUT))} configs in {OUT}")
