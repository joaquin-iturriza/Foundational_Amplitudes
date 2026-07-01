#!/usr/bin/env python
"""Emit the 3 DyHPO sweep configs for the coupling+mass A/B (off/scalar/diagram),
mirroring the established encab sweep pattern (sweep_config_jeanzay_encab_*.yaml):
identical recipe + base HPs + seed across arms, only the coupling/mass FEATURE
flags differ, and DyHPO optimises the FULL HP space (lr, regularization_lambda,
warmup, eta_min, ema_decay) per arm. Best-vs-best on val_loss_no_reg.

Writes sweep/sweep_config_jeanzay_scan_ab_<arm>.yaml.
"""
import os, yaml

PROJ = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"

ARMS = {
    "off":     {"data.mass_from_momenta": "false", "data.coupling_scalars": "false",
                "model.use_diagrams": "false"},
    "scalar":  {"data.mass_from_momenta": "true",  "data.coupling_scalars": "true",
                "model.use_diagrams": "false"},
    "diagram": {"data.mass_from_momenta": "true",  "data.coupling_scalars": "false",
                "model.use_diagrams": "true", "model.d_diag": 32},
    # scalar arm + the INTERNAL-mass propagator off-shellness (s−M_Z², direct feed).
    # Isolates the marginal value of the internal-mass lever on top of scalar:
    # best(offshell) vs best(scalar) on val_loss_no_reg. Uses the Z-mass-scan datasets
    # in scan_bigrun.yaml where s−M_Z² is genuinely new info (M_Z hidden, s-channel).
    "offshell": {"data.mass_from_momenta": "true", "data.coupling_scalars": "true",
                 "model.use_diagrams": "false",
                 "data.internal_mass_scalars": "true", "data.offshell_per_event": "true",
                 "data.internal_mass_pdgs": "[23]"},
}

BASE_FIXED = {
    # recipe data (already prebuilt -> require_cache, no prepost prebuild needed)
    "data.source": "recipes",
    "data.processes_file": f"{PROJ}/recipes/scan_bigrun.yaml",
    "data.require_cache": "true",
    "data.train_subsample": 2000,
    "data.eval_subsample": 500,
    "data.preprocess_per_dataset": "true",
    "data.seed": 42,
    "data.use_PIDs": "false",
    "data.spin_onehot": "true",
    "data.color_onehot": "true",
    "data.prop_is_massless": "true",
    "data.standardize_props": "true",
    "model": "lloca",
    "seed": 42,
    "training.batchsize": 1024,
    "evaluation.batchsize": 4096,
    "training.loss_aggregation": "geometric_mean",
    "training.regularization": "L2",
    "training.scheduler": "CosineAnnealingLR",
    "training.get_ID": "false",
    "training.save_intermediate": "false",
    "training.validate_every_n_steps": 500,
    "training.dtype": "float32",
    "plot": "true",
    "use_mlflow": "false",
}

# Full HP search space — the same one the encab A/B sweeps use.
SEARCH_SPACE = [
    {"name": "training.lr", "type": "float_log", "low": 1.0e-4, "high": 1.0e-2},
    {"name": "training.regularization_lambda", "type": "float_log", "low": 1.0e-11, "high": 1.0e-6},
    {"name": "training.cosanneal_warmup_frac", "type": "float_uniform", "low": 0.0, "high": 0.2},
    {"name": "training.cosanneal_eta_min", "type": "float_log", "low": 1.0e-11, "high": 1.0e-6},
    {"name": "training.ema_decay", "type": "float_uniform", "low": 0.9, "high": 0.9999},
]


def make(arm, feat):
    return {
        "cluster": {"account": "itg@v100", "partition": "gpu_p2", "request_gpus": 1,
                    "cpus_per_task": 8, "scheduler": "slurm", "time": "06:00:00",
                    "auto_submit": False},
        "dyhpo": {"n_candidates": 300, "n_startup": 8, "seed": 42, "total_budget": 10000},
        # Single fidelity: every trial runs the full 5000 steps so a slow-converging
        # arm (diagram only pulls ahead ~5k steps) is screened fairly — no
        # multi-fidelity early-stopping bias against it. Pure BO over the HP space.
        "fidelity_schedule": {"t_steps": [5000]},
        "n_trials": 15,
        "fixed_params": {**BASE_FIXED, **feat},
        "paths": {
            "project_dir": PROJ,
            "sweep_dir": f"{PROJ}/sweeps/scan_ab",
            "setup_commands": [
                "module load anaconda-py3/2023.09",
                "source /gpfslocalsup/pub/anaconda-py3/2023.09/etc/profile.d/conda.sh",
                "conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational",
            ],
        },
        "sweep_name": f"scan_ab_{arm}",
        "range_extension": {"enabled": False},
        "search_space": SEARCH_SPACE,
    }


for arm, feat in ARMS.items():
    out = f"{PROJ}/sweep/sweep_config_jeanzay_scan_ab_{arm}.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(make(arm, feat), f, sort_keys=False)
    print("wrote", out)
