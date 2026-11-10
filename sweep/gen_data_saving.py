"""pretrain_data_saving stage 2: single-fidelity 8-trial DyHPO configs, one per (target, D, arm), for the three held-out
tree targets of catalog_v4 (ee->ttbar, ee->ccbar g, ee->bbbar gg), from scratch and fine-tuned.

Every trial starts from a parent run's saved config.yaml (base_config, sweep/run_trial.py), so the model and inputs are
the parent's and only the keys below and the searched HPs move:
  scratch    base_config = the parent of the calibration (trial 1262 of cov_v4_long_hpo8_cc_68224), no pretrained
             weights, the target's first D train events (data.train_subsample) of recipes/catalog_v4_datasaving_<t>.yaml
  fine-tune  base_config = the family-held-out pretraining (--ft-parent, kit_benchmark), full retrain from its best
             checkpoint with layer decay (fine_tune.lr_scale [0.1, 10], fine_tune.layer_decay [0.75, 1]); the target's
             own amplitude stats, the parent's off-shellness scale (as scripts/job_data_saving.sh ft)
Both: training.heterosc_select=mu_mse (DyHPO observes the val mu MSE at the best checkpoint, base_experiment.py
1182-1304), the sigma stage after, data.phase_space_on_pool, the fixed 1e4 val split, seed 42, warm_start_idx=null.
lr window (scratch): one decade either side, [c/10, 10c], of c = 7.82114e-3 (T/17056)^-0.55 times the transfer
study's D factor: the lr the calibration ran at (scripts/job_data_saving.sh, the design passed 2026-11-10 03:35),
the parent's MuonAdamW law (CLAUDE.md: a Muon sweep centres on its own search; the AdamW-measured lr surface of
gen_transfer_pilot.lr_space sits ~10x lower at T=8000, 9.6e-4 at k=2). The window's width is sec:ladder's one decade
either side; the parent's law is censored at its own window's top (a lower bound), hence 10c above. The calibration
put ee->ccbar g and ee->bbbar gg's best inside the warm-up at c (analysis/data_saving/data/horizons_cal2.json), hence
c/10 below.
The other searched HPs are the template's (sweep/sweep_config_jeanzay_template.yaml).

Horizons: --phase a writes the scratch sweeps of ee->ccbar g and ee->bbbar gg at the calibration horizon (8000 steps
k <= 8, 12000 k = 9, 10); --phase b writes both arms at T(D) (ee->ttbar's from the calibration, the others' from
phase a's best trials, --horizons JSON {target: {k: T}}). Wall limits from the calibration's measured step rates.

    python sweep/gen_data_saving.py --phase a
    python sweep/gen_data_saving.py --phase b --ft-parent <run dir on the site> [--horizons file.json] [--targets ...]
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
TARGETS = ["epem_ttx", "epem_ccxg", "epem_bbxgg"]
SCR_PARENT = "runs/catalog_v4_long/best_cov_v4_long_hpo8_cc_68224_trial_1262"
CAL = os.path.join(ROOT, "analysis", "data_saving", "data", "horizons_cal2.json")
N_TRIALS, N_STARTUP = 8, 3
OVERHEAD_H = 0.08                       # data load, test pass, sigma stage per trial (calibration runs)
T_CAL = {k: (8000 if k <= 8 else 12000) for k in range(2, 11)}


def d_of(k):
    return round(10 ** (k / 2))


def lr_centre(T, k):
    """The calibration's lr (scripts/job_data_saving.sh): the parent's law at T times the D factor."""
    from sweep.gen_transfer_pilot import D_FACTOR
    Ns = np.array(sorted(D_FACTOR)); B = np.array([D_FACTOR[n] for n in Ns])
    D = d_of(k)
    return 7.82114e-3 * (T / 17056) ** -0.55 * float(np.exp(np.interp(np.log(np.clip(D, Ns.min(), Ns.max())),
                                                                       np.log(Ns), B)))


def rates():
    """{(target, k): s/step} measured in the calibration (main stage, validation included)."""
    return {(r["target"], r["k"]): r["rate_s_per_step"] for r in json.load(open(CAL)) if r.get("rate_s_per_step")}


def ttx_horizons():
    return {r["k"]: r["T"] for r in json.load(open(CAL)) if r["target"] == "epem_ttx" and r.get("T")}


def template_space():
    t = yaml.safe_load(open(os.path.join(HERE, "sweep_config_jeanzay_template.yaml")))
    return [s for s in t["search_space"] if s["name"] != "training.lr"]


def write(name, tgt, k, T, arm, rate, ft_parent=None):
    c = lr_centre(T, k)
    space = template_space()
    fixed = {"data.source": "recipes", "data.processes_file": f"${{PROJECT_DIR}}/recipes/catalog_v4_datasaving_{tgt}.yaml",
             "data.train_subsample": d_of(k), "data.eval_subsample": "null", "data.phase_space_on_pool": "true",
             "data.memmap": "false", "training.heterosc_select": "mu_mse", "evaluation.train_subsample": 2000,
             "seed": 42, "plot": "true"}
    if arm == "scr":
        base = f"${{PROJECT_DIR}}/{SCR_PARENT}"
        fixed["fine_tune.pretrained_path"] = "null"
        space = [{"name": "training.lr", "type": "float_log", "low": float(f"{c / 10:.3g}"),
                  "high": float(f"{10 * c:.3g}")}] + space
    else:
        base = ft_parent
        fixed.update({"fine_tune.pretrained_path": f"{ft_parent}/models/model_run0_best.pt",
                      "fine_tune.target_stats": "own", "fine_tune.offshell_stats": "parent"})
        space = [{"name": "fine_tune.lr_scale", "type": "float_log", "low": 0.1, "high": 10.0},
                 {"name": "fine_tune.layer_decay", "type": "float_uniform", "low": 0.75, "high": 1.0}] + space
    hours = T * rate / 3600 + OVERHEAD_H
    cfg = {"cluster": {"scheduler": "slurm", "auto_submit": False, "request_gpus": 1, "mem": "32G",
                       "cpus_per_task": 4,
                       # the estimate itself: sweep/drive.py applies the one margin (1.5x + 6 min)
                       "time": "%02d:%02d:00" % divmod(max(1, int(round(hours * 60))), 60)},
           "paths": None, "sweep_name": name, "base_config": base, "n_trials": N_TRIALS,
           "dyhpo": {"surrogate": "gp", "n_candidates": 2000, "objective_scale": "auto",
                     "seed": zlib.crc32(name.encode()) % 2 ** 31, "n_startup": N_STARTUP, "total_budget": 10000},
           "fidelity_schedule": {"t_steps": [int(T)]}, "fixed_params": fixed,
           "range_extension": {"enabled": False}, "search_space": space}
    path = os.path.join(HERE, f"sweep_config_{name}.yaml")
    with open(path, "w") as f:
        f.write(f"# pretrain_data_saving stage 2: {tgt} D={d_of(k)} {arm} T={T}; est {hours:.2f} GPU-h per trial."
                "\n# Written by sweep/gen_data_saving.py.\n" + yaml.safe_dump(cfg, sort_keys=False))
    return path, hours * N_TRIALS


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=["a", "b"], required=True)
    ap.add_argument("--ft-parent", help="the fine-tune parent's run dir on the site (phase b)")
    ap.add_argument("--horizons", help="JSON {target: {k: T}} for ee->ccbar g and ee->bbbar gg (phase b)")
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    ap.add_argument("--arms", nargs="*", default=["scr", "ft"])
    a = ap.parse_args()
    R = rates()
    jobs = []
    if a.phase == "a":
        for t in [x for x in a.targets if x != "epem_ttx"]:
            for k in range(2, 11):
                jobs.append((f"ds_scr_{t}_d{k}_t{T_CAL[k]}", t, k, T_CAL[k], "scr"))
    else:
        hz = {"epem_ttx": ttx_horizons()}
        if a.horizons:
            hz.update({t: {int(k): v for k, v in d.items()} for t, d in json.load(open(a.horizons)).items()})
        for t in a.targets:
            for k in range(2, 11):
                T = hz.get(t, {}).get(k)
                if T is None:
                    continue
                for arm in a.arms:
                    if arm == "scr" and t != "epem_ttx" and T == T_CAL[k]:
                        continue                 # phase a's sweep at this horizon is the scratch arm
                    if arm == "ft" and not a.ft_parent:
                        raise SystemExit("--ft-parent is needed for the fine-tune arm")
                    jobs.append((f"ds_{arm}_{t}_d{k}_t{T}", t, k, T, arm))
    miss = sorted({(t, k) for _, t, k, _, _ in jobs if (t, k) not in R})
    if miss:
        raise SystemExit(f"no measured step rate for {miss} in {CAL}: rerun analysis/data_saving/horizons.py first")
    total = 0.0
    for name, t, k, T, arm in jobs:
        rate = R[(t, k)]
        p, h = write(name, t, k, T, arm, rate, a.ft_parent)
        total += h
        print(f"{os.path.relpath(p, ROOT)}  {h:.1f} GPU-h")
    print(f"{len(jobs)} sweeps, {total:.1f} GPU-h estimated (8 trials each)")


if __name__ == "__main__":
    main()
