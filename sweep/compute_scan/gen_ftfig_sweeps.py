"""Emit per-cell HPO sweep configs for the ftfig families (ft416raw/ft416best/ft352lo),
mirroring the cscan_ft25_D*.yaml sweeps exactly (same search space, 10 trials/level,
same t-grids per D) — only the pretrained checkpoint and the encoding fixed_params
differ per family (flags copied from analysis/scaling_compute/gen_ftfig_cells.py).
"""
import copy, os, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes"

RAW = {"data.spin_onehot": "false", "data.color_onehot": "false",
       "data.prop_is_massless": "false", "data.standardize_props": "false",
       "data.mass_from_momenta": "false", "data.coupling_scalars": "false",
       "data.internal_mass_scalars": "false", "data.offshell_per_event": "false"}
BEST = {"data.spin_onehot": "true", "data.color_onehot": "true",
        "data.prop_is_massless": "true", "data.standardize_props": "true",
        "data.mass_from_momenta": "true", "data.coupling_scalars": "true",
        "data.internal_mass_scalars": "true", "data.offshell_per_event": "true",
        "data.internal_mass_pdgs": "[23,6,25]", "data.amp_orders": "[[1,0]]"}
FAMS = {"ft416raw": ("raw416", RAW), "ft416best": ("best416", BEST),
        "ft352lo": ("lo352", BEST)}
RUNG2 = dict(RAW, **{"data.spin_onehot": "true", "data.color_onehot": "true",
                     "data.prop_is_massless": "true", "data.standardize_props": "true"})
RUNG3 = dict(RUNG2, **{"data.mass_from_momenta": "true", "data.coupling_scalars": "true"})
# ladder rungs: one sweep per (rung, target) at D=100k, t=6193 only — all parallel
LADDER = {"ftraw1h": ("raw1h", RAW), "ftrung2": ("rung2_1h", RUNG2),
          "ftrung3": ("rung3_1h", RUNG3), "ftbest1h": ("best1h", BEST)}

# narrowed per the settled HPO search-space rules (CLAUDE.md / results.tex):
# lr_scale [0.1,10]@1, layer_decay [0.75,1.0], reg capped 1e-6, warmup [0.05,0.2],
# eta_min fixed ~1e-8 (weakest knob — dropped from the space).
SEARCH = [
    {"name": "fine_tune.lr_scale", "type": "float_log", "low": 0.1, "high": 10.0},
    {"name": "fine_tune.layer_decay", "type": "float_uniform", "low": 0.75, "high": 1.0},
    {"name": "training.regularization_lambda", "type": "float_log",
     "low": 1.0e-10, "high": 1.0e-06},
    {"name": "training.cosanneal_warmup_frac", "type": "float_uniform",
     "low": 0.05, "high": 0.2},
]
COMMON = {"data.use_PIDs": "false", "data.preprocess_per_dataset": "true",
          "data.seed": 42, "model.use_diagrams": "false",
          "model.particle_encoder_hidden": 0}

out = []
for D in ("10k", "100k", "1M"):
    src = f"{HERE}/cscan_ft25_D{D}.yaml"
    if not os.path.exists(src):  # ft25 D100k config is the _002 rerun; same name here
        raise SystemExit(f"missing template {src}")
    base = yaml.safe_load(open(src))
    for fam, (arm, flags) in FAMS.items():
        cfg = copy.deepcopy(base)
        cfg["sweep_name"] = f"cscan_{fam}_D{D}_hpo"
        cfg["n_trials_per_level"] = 8
        cfg["search_space"] = copy.deepcopy(SEARCH)
        fp = cfg["fixed_params"]
        fp["training.cosanneal_eta_min"] = 1.0e-08
        fp["fine_tune.pretrained_path"] = (
            f"{ROOT}/compare_models/_ftfig_pre_{arm}/models/model_run0_best.pt.gz")
        fp.update(COMMON)
        fp.update(flags)
        dst = f"{HERE}/cscan_{fam}_D{D}_hpo.yaml"
        yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False, default_flow_style=False)
        out.append(dst)
base100k = yaml.safe_load(open(f"{HERE}/cscan_ft25_D100k.yaml"))
for fam, (arm, flags) in LADDER.items():
    cfg = copy.deepcopy(base100k)
    cfg["sweep_name"] = f"cscan_{fam}_hpo"
    cfg["t_steps_values"] = [6193]
    cfg["n_trials_per_level"] = 8
    cfg["search_space"] = copy.deepcopy(SEARCH)
    fp = cfg["fixed_params"]
    fp["training.cosanneal_eta_min"] = 1.0e-08
    fp["fine_tune.pretrained_path"] = (
        f"{ROOT}/compare_models/_ftfig_pre_{arm}/models/model_run0_best.pt.gz")
    fp.update(COMMON)
    fp.update(flags)
    dst = f"{HERE}/cscan_{fam}_hpo.yaml"
    yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False, default_flow_style=False)
    out.append(dst)

print(f"wrote {len(out)} configs:")
for p in out:
    print(" ", os.path.basename(p))
