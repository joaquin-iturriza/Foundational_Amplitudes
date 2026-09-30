"""Transfer study, pilot (chapter 1): pretrain on ee->uu alone, then the data-scaling curve of
two probes, a near one (ee->dd~) and a far one (uu~->gg), from scratch and fine-tuned.

  pretrain   tp_pre_ee_uu              ee_uu, 100k train events, T_PRE steps
  scratch    tp_scr_<probe>_d<k>       probe alone on the first D train events
  fine-tune  tp_ft_<probe>_d<k>        the same, from the pretrain's best checkpoint

D = 10^(k/2), k = 2..10 (10 ... 1e5), prefixes of one 100k pool (the pool is shuffled, so
each prefix is a random subset and the smaller sets are nested in the larger); val and test
are the fixed 10k splits at every D. What depends on the momenta alone (momentum scale, which
propagator factors the target divides out) is fitted on the whole 100k pool's phase-space
points, which carry no labels; the amplitude transform and mean/std are fitted on the D train
events (data.phase_space_on_pool). Every cell trains on one target, and its loss converts to
one unit, MSE of log|M|^2 = val_loss_no_reg * prepd_std^2 (data_stats.json). The train batch is
B = min(16384, D/2) (experiment.py caps it at half the train split): the batch moves with D.
One single-fidelity DyHPO per cell, N_TRIALS trials, one seed (the user's call for the pilot:
see how noisy a single seed is before adding more).

The horizon is deliberately generous (T_CELL): the pilot measures where each D stops
improving (the best checkpoint's step), and that sets the horizons of the full study.
Search spaces are wide, at the user's request for this study (2026-09-30: "general and wide,
like the ones we were doing before", to be narrowed once patterns show), so the lr window is
one decade either side instead of CLAUDE.md's half decade:
  scratch / pretrain  lr one decade either side of lr*(t, high D) (analysis/hpo_optima, measured
                      at batch 16384) * sqrt(B/16384), Adam's square-root batch rule (the user's
                      choice, 2026-09-30); the template's lambda, warm-up, eta_min, EMA
  fine-tune           training.lr = the pretrain's best * sqrt(B/16384), fine_tune.lr_scale [0.1, 10],
                      fine_tune.layer_decay [0.75, 1], lambda, warm-up, eta_min, EMA;
                      amplitude stats fitted on the probe (fine_tune.target_stats=own)
The fine-tune configs need the pretrain's best checkpoint:
    python sweep/gen_transfer_pilot.py                      pretrain + scratch configs
    python sweep/gen_transfer_pilot.py --ft CKPT --lr LR    fine-tune configs
"""
import argparse, collections, json, os
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PRE, PROBES = "ee_uu", ["ee_ddbar", "uubar_gg"]
KS = range(2, 11)                                  # D = 10^(k/2)
T_PRE = 16000
T_CELL = {k: (2000 if k <= 5 else 4000 if k <= 7 else 8000) for k in KS}
N_TRIALS, N_STARTUP = 8, 3
SEC_PER_STEP, OVERHEAD_MIN = 0.45, 15              # bs 16384 on a V100 (gen_solo16k_configs)

recs = json.load(open(os.path.join(ROOT, "analysis", "hpo_optima", "hpo_optima.json")))
row = collections.defaultdict(list)
for r in recs:
    if r["family"] == "scaling_p" and r["converged"] and r["hp_best"].get("training.lr") and r["n_train"] == 70000:
        row[r["t_steps"]].append(np.log(r["hp_best"]["training.lr"]["val"]))
ts = np.array(sorted(row)); lr_row = np.array([np.mean(row[t]) for t in ts])
lr_star = lambda t: float(np.exp(np.interp(np.log(t), np.log(ts), lr_row)))

FIXED = {
    "data.source": "recipes", "data.require_cache": "false", "data.eval_subsample": 10000,
    "data.phase_space_on_pool": "true",
    "data.preprocess_per_dataset": "true", "data.signedlog_quantile": 0.01, "data.seed": 42,
    "data.use_PIDs": "false", "data.spin_onehot": "true", "data.color_onehot": "true",
    "data.prop_is_massless": "true", "data.standardize_props": "true",
    "data.generation_onehot": "true", "data.generation_feature": "true",
    "data.mass_from_momenta": "false", "data.coupling_scalars": "true",
    "data.internal_mass_scalars": "true", "data.offshell_per_event": "true",
    "data.target_propagators": "true", "data.target_propagator_tchannel": "true",
    "data.target_propagator_tchannel_max_final": 2, "data.internal_mass_pdgs": "[23,6,25]",
    "model": "lloca", "model.use_diagrams": "false", "model.particle_encoder_hidden": 0,
    "model.net.num_heads": 8, "model.net.num_blocks": 8, "seed": 42,
    "training.batchsize": 16384, "evaluation.batchsize": 16384, "evaluation.train_subsample": 2000,
    "training.loss_aggregation": "mean", "training.sign_head": "true",
    "training.val_aggregation": "geometric_mean", "training.regularization": "L2",
    "training.scheduler": "CosineAnnealingLR", "training.get_ID": "false",
    "training.save_intermediate": "false", "training.validate_frac": "0.01",
    "training.es_load_best_model": "true", "training.dtype": "float32",
    "plot": "true", "use_mlflow": "false",
}
COMMON_SPACE = [
    {"name": "training.regularization_lambda", "type": "float_log", "low": 1.0e-10, "high": 1.0e-6},
    {"name": "training.cosanneal_warmup_frac", "type": "float_uniform", "low": 0.05, "high": 0.2},
    {"name": "training.cosanneal_eta_min", "type": "float_log", "low": 1.0e-10, "high": 1.0e-7},
    {"name": "ema", "type": "categorical", "choices": ["false", "true"]},
    {"name": "training.ema_decay", "type": "float_uniform", "low": 0.9, "high": 0.999},
]


def batch(k):
    return int(min(16384, round(10 ** (k / 2)) / 2))


def lr_space(T, k=None):
    c = lr_star(T) * (np.sqrt(batch(k) / 16384) if k is not None else 1.0)
    return {"name": "training.lr", "type": "float_log",
            "low": float(f"{c / 10:.3g}"), "high": float(f"{c * 10:.3g}")}


def write(name, recipe, T, space, extra=None, head=""):
    fixed = dict(FIXED, **{"data.processes_file": f"${{PROJECT_DIR}}/recipes/{recipe}"}, **(extra or {}))
    c = {"cluster": {"scheduler": "slurm", "auto_submit": False, "request_gpus": 1, "mem": "8G",
                     "cpus_per_task": 4,
                     "time": "%02d:%02d:00" % divmod(int(T * SEC_PER_STEP / 60 * 1.5 + OVERHEAD_MIN), 60)},
         "paths": None, "sweep_name": name, "n_trials": N_TRIALS,
         "dyhpo": {"n_candidates": 200, "seed": 42, "n_startup": N_STARTUP, "total_budget": 10000},
         "fidelity_schedule": {"t_steps": [T]}, "fixed_params": fixed,
         "range_extension": {"enabled": False}, "search_space": space}
    path = os.path.join(HERE, f"sweep_config_{name}.yaml")
    open(path, "w").write(f"# {head}\n# Written by sweep/gen_transfer_pilot.py.\n" + yaml.safe_dump(c, sort_keys=False))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", help="best checkpoint of the pretrain (absolute path on the site)")
    ap.add_argument("--lr", type=float, help="the pretrain's best training.lr")
    a = ap.parse_args()
    out = []
    if a.ft is None:
        out.append(write(f"tp_pre_{PRE}", f"ref_solo_{PRE}.yaml", T_PRE, [lr_space(T_PRE)] + COMMON_SPACE,
                         head=f"Transfer pilot pretrain: {PRE} alone, 100k events, {T_PRE} steps."))
        for p in PROBES:
            for k in KS:
                out.append(write(f"tp_scr_{p}_d{k}", f"transfer_probe_{p}.yaml", T_CELL[k],
                                 [lr_space(T_CELL[k], k)] + COMMON_SPACE,
                                 {"data.train_subsample": int(round(10 ** (k / 2)))},
                                 head=f"Transfer pilot, scratch: {p} on D = 10^{k / 2:g} events."))
    else:
        assert a.lr, "--lr (the pretrain's best lr) is needed with --ft"
        ft_space = [{"name": "fine_tune.lr_scale", "type": "float_log", "low": 0.1, "high": 10.0},
                    {"name": "fine_tune.layer_decay", "type": "float_uniform", "low": 0.75, "high": 1.0}]
        for p in PROBES:
            for k in KS:
                out.append(write(f"tp_ft_{p}_d{k}", f"transfer_probe_{p}.yaml", T_CELL[k], ft_space + COMMON_SPACE,
                                 {"data.train_subsample": int(round(10 ** (k / 2))), "training.lr": float(f"{a.lr * np.sqrt(batch(k) / 16384):.3g}"),
                                  "fine_tune.pretrained_path": a.ft, "fine_tune.target_stats": "own"},
                                 head=f"Transfer pilot, fine-tune from {PRE}: {p} on D = 10^{k / 2:g} events."))
    print("\n".join(os.path.relpath(p, ROOT) for p in out))
    print("lr* centres:", {T: f"{lr_star(T):.2g}" for T in sorted({T_PRE, *T_CELL.values()})})


if __name__ == "__main__":
    main()
