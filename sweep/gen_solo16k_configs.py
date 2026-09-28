"""Emit the batch-16384 solo references: the 12 reference processes of the steps curve alone on their full
catalog pools at the canonical batch, 63 ... 4000 steps (the joint horizons without 8000, and 63 / 125 / 250 below them so the
floor-aware fit has points before the curves bend),
one single-fidelity DyHPO sweep per (steps, process), 6 trials. Written from the batch-1024 configs
(sweep/sweep_config_solob1k_t33_<process>.yaml: recipe, encoding, target levers, as the joint steps runs)
with the batch at 16384 and the search reduced to the knobs that mattered there
(analysis/catalog_v2/solo_b1k_inspect.py: lr 34% of the within-sweep variance, warm-up 3%, EMA 1%,
lambda 0.2%, eta_min 0.1%):
  training.lr                    +-1/2 decade around lr*(t, D) (CLAUDE.md HPO rule 2), the high-D row
                                 of the measured surface (analysis/hpo_optima/hpo_optima.json, scaling_p,
                                 D = 70000) interpolated log-log in t
  training.cosanneal_warmup_frac [0.05, 0.2]
  fixed: cosanneal_eta_min 0, regularization_lambda 1e-8 (job_catalog_short_ab.sh default), ema off.
    python sweep/gen_solo16k_configs.py            writes sweep/sweep_config_solo16k_t<T>_<process>.yaml
"""
import collections, glob, json, os, re
import numpy as np
import yaml
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
STEPS = [63, 125, 250, 500, 1000, 2000, 4000]
PROCS = ["ee_aa", "uubar_uubar", "ee_uu", "ee_ddbar", "ee_uug", "udbar_WpZZ", "ee_uugg", "udbar_WpZaa",
         "uubar_ZaZ_nlo", "ee_bb_nlo", "udbar_Wgg_nlo", "uubar_ddbara_nlo"]
HIGH_D = 70000
SEC_PER_STEP, OVERHEAD_MIN = 0.45, 15          # the joint steps runs at this batch: 0.42-0.44 s/step

recs = json.load(open(os.path.join(ROOT, "analysis", "hpo_optima", "hpo_optima.json")))
row = collections.defaultdict(list)
for r in recs:
    if r["family"] == "scaling_p" and r["converged"] and r["hp_best"].get("training.lr") and r["n_train"] == HIGH_D:
        row[r["t_steps"]].append(np.log(r["hp_best"]["training.lr"]["val"]))
ts = np.array(sorted(row)); lr_row = np.array([np.mean(row[t]) for t in ts])
lr_star = lambda t: float(np.exp(np.interp(np.log(t), np.log(ts), lr_row)))

for p in PROCS:
    base = yaml.safe_load(open(os.path.join(HERE, f"sweep_config_solob1k_t33_{p}.yaml")))
    for T in STEPS:
        c = dict(base); c["fixed_params"] = dict(base["fixed_params"])
        c["sweep_name"] = f"solo16k_t{T}_{p}"
        c["fidelity_schedule"] = {"t_steps": [T]}
        c["cluster"] = dict(base["cluster"], mem="8G",   # measured peak RSS 1.7 GB at this batch
                            time="%02d:%02d:00" % divmod(int(T * SEC_PER_STEP / 60 * 1.5 + OVERHEAD_MIN), 60))
        c["fixed_params"].update({"training.batchsize": 16384, "evaluation.batchsize": 16384,
                                  "training.cosanneal_eta_min": 0.0, "training.regularization_lambda": 1.0e-8,
                                  "ema": "false"})
        centre = lr_star(T)
        c["search_space"] = [
            {"name": "training.lr", "type": "float_log", "low": float(f"{centre / 10 ** 0.5:.3g}"), "high": float(f"{centre * 10 ** 0.5:.3g}")},
            {"name": "training.cosanneal_warmup_frac", "type": "float_uniform", "low": 0.05, "high": 0.2}]
        head = (f"# Solo reference at the canonical batch: {p} alone on its full catalog pool, bs 16384, {T} steps.\n"
                f"# Written by sweep/gen_solo16k_configs.py (lr window +-1/2 decade around lr*(t={T}, high D) = {centre:.2g}).\n")
        open(os.path.join(HERE, f"sweep_config_solo16k_t{T}_{p}.yaml"), "w").write(head + yaml.safe_dump(c, sort_keys=False))
    print(p, "done")
print("lr* centres:", {T: f"{lr_star(T):.2g}" for T in STEPS})
