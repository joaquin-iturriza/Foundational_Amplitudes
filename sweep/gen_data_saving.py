"""pretrain_data_saving stage 2: single-fidelity 8-trial DyHPO configs, one per (target, D, arm), for the draft Fig. 22
targets (D385-D387): e+e- > u u~ and e+e- > t t~ at LO, and their one-loop V (e+e- > u u~ V trained on V, D375;
e+e- > t t~ V on V/B, physics factors off), from scratch and fine-tuned.

Both arms of a target start from its parent's saved config.yaml (scripts/job_datasaving_parent.sh, the D392 input
flags in): the LO targets from the held-out parent (--parent-lo: the four targets' equality classes out), the one-loop
targets from the trees-in parent (--parent-nlo: the same plus e+e- > u u~ and t t~ LO, the entry's "the one loops with
the tree levels in the pretraining"), so model and inputs match and only the keys below and the searched HPs move:
  scratch    no pretrained weights; training.lr searched over [c/3, 3c] (HPO rule 2: half a decade either side)
  fine-tune  full retrain from the parent's final model (models/model_run0_best.pt, which the sigma stage, the end of
             the parent's training, leaves as its final model, D276) with layer decay (fine_tune.lr_scale [0.1, 10] around
             training.lr = c, fine_tune.layer_decay [0.75, 1]); the target's own amplitude stats, the parent's
             off-shellness scale (as scripts/job_data_saving.sh ft)
c = 7.82114e-3 (max(T,3000)/17056)^-0.55 min(1, T/3000)^0.5 times the transfer study's D factor (the calibration's law,
job_data_saving.sh, with HPO rule 2's inverted U below t* = 3000 steps), halved where the calibration cell trained at half lr.
Both: the target's first D train events (data.train_subsample) of recipes/catalog_v4_datasaving_<t>.yaml,
training.heterosc_select=mu_mse, the sigma stage after, data.phase_space_on_pool, the frozen val split, seed 42.
The other searched HPs are the template's (sweep/sweep_config_jeanzay_template.yaml).
Horizons T(D) and step rates: analysis/data_saving/horizons.py --ladder outputs (D391: 1.1 x the validation minimum,
non-decreasing, cap 8500; fine-tunes the same steps, D383).

    python sweep/gen_data_saving.py --parent-lo <run dir> --parent-nlo <run dir> [--targets ...] [--arms scr ft]
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
TARGETS = ["ee_uu", "epem_ttx", "ee_uu_nlo", "ee_ttbar_nlo"]
CALS = [os.path.join(ROOT, "analysis", "data_saving", "data", f) for f in
        ("horizons_cal_f22_cal_f22_lrhalf.json", "horizons_cal2_ladder.json")]
VB = {"ee_ttbar_nlo"}                   # trained on V/B (D375), physics factors off (job_data_saving.sh)
N_TRIALS, N_STARTUP = 8, 3
OVERHEAD_H = 0.08                       # data load, test pass, sigma stage per trial (calibration runs)


def d_of(k):
    return round(10 ** (k / 2))


def lr_centre(T, k):
    """The lr centre: the calibration's law (scripts/job_data_saving.sh), the parent's law at T times the D factor. Below the optimizer
    timescale t* = 3000 steps the law's t^-0.55 branch does not hold: lr* rises as (T/3000)^0.5 up to t* (HPO rule 2,
    the inverted U), so a horizon T < 3000 takes the law at 3000 times (T/3000)^0.5."""
    from sweep.gen_transfer_pilot import D_FACTOR
    Ns = np.array(sorted(D_FACTOR)); B = np.array([D_FACTOR[n] for n in Ns])
    D = d_of(k)
    return 7.82114e-3 * (max(T, 3000) / 17056) ** -0.55 * min(1.0, T / 3000) ** 0.5 * float(np.exp(np.interp(np.log(np.clip(D, Ns.min(), Ns.max())),
                                                                       np.log(Ns), B)))


def cal_rows():
    return [r for f in CALS for r in json.load(open(f)) if r["target"] in TARGETS]


def rates():
    """{(target, k): s/step} measured in the calibration (main stage, validation included)."""
    return {(r["target"], r["k"]): r["rate_s_per_step"] for r in cal_rows() if r.get("rate_s_per_step")}


def horizons():
    """{target: {k: T(D)}} of the D391 rule (horizons.py --ladder)."""
    out = {}
    for r in cal_rows():
        if r.get("T_D"):
            out.setdefault(r["target"], {})[r["k"]] = r["T_D"]
    return out


def template_space():
    t = yaml.safe_load(open(os.path.join(HERE, "sweep_config_jeanzay_template.yaml")))
    return [s for s in t["search_space"] if s["name"] != "training.lr"]


def halved():
    """{(target, k)} whose T(D) and rate come from a rerun at half the calibration lr (the cell blew up at full lr), so
    the search centres on the lr that cell trained at."""
    return {(r["target"], r["k"]) for r in cal_rows() if r.get("T_D") and str(r.get("dir", "")).endswith("_lrhalf")}


def write(name, tgt, k, T, arm, rate, parent, lr_factor=1.0):
    c = lr_centre(T, k) * lr_factor
    space = template_space()
    fixed = {"data.source": "recipes", "data.processes_file": f"${{PROJECT_DIR}}/recipes/catalog_v4_datasaving_{tgt}.yaml",
             "data.train_subsample": d_of(k), "data.eval_subsample": "null", "data.phase_space_on_pool": "true",
             "data.memmap": "false", "training.heterosc_select": "mu_mse", "evaluation.train_subsample": 2000,
             "seed": 42, "plot": "true", "warm_start_idx": "null"}
    if tgt in VB:
        fixed.update({"data.target_over_born": "true", "data.target_physics_factor": "false",
                      "data.target_physics_orders": "false"})
    base = parent
    if arm == "scr":
        fixed["fine_tune.pretrained_path"] = "null"
        space = [{"name": "training.lr", "type": "float_log", "low": float(f"{c / 3:.3g}"),
                  "high": float(f"{3 * c:.3g}")}] + space
    else:
        fixed.update({"training.lr": float(f"{c:.4g}"), "fine_tune.pretrained_path": f"{parent}/models/model_run0_best.pt",
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
                     "seed": zlib.crc32(name.encode()) % 2 ** 31, "n_startup": N_STARTUP, "total_budget": 10000,
                     # D244/D245: the start-up trials, then two sequential batches over the rest
                     "waves": [N_STARTUP, (N_TRIALS - N_STARTUP + 1) // 2, (N_TRIALS - N_STARTUP) // 2]},
           "fidelity_schedule": {"t_steps": [int(T)]}, "fixed_params": fixed,
           "range_extension": {"enabled": False}, "search_space": space}
    path = os.path.join(HERE, f"sweep_config_{name}.yaml")
    with open(path, "w") as f:
        f.write(f"# pretrain_data_saving stage 2: {tgt} D={d_of(k)} {arm} T={T}; est {hours:.2f} GPU-h per trial."
                "\n# Written by sweep/gen_data_saving.py.\n" + yaml.safe_dump(cfg, sort_keys=False))
    return path, hours * N_TRIALS


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parent-lo", required=True, help="the held-out parent's run dir on the site")
    ap.add_argument("--parent-nlo", required=True, help="the trees-in parent's run dir on the site")
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    ap.add_argument("--arms", nargs="*", default=["scr", "ft"])
    a = ap.parse_args()
    R, hz = rates(), horizons()
    jobs = []
    for t in a.targets:
        for k in range(2, 10 if t.endswith("_nlo") else 11):
            T = hz.get(t, {}).get(k)
            if T is None:
                raise SystemExit(f"no T(D) for {t} k={k}: rerun analysis/data_saving/horizons.py --ladder")
            for arm in a.arms:
                jobs.append((f"dsave_s2_{arm}_{t}_d{k}_t{T}", t, k, T, arm))
    miss = sorted({(t, k) for _, t, k, _, _ in jobs if (t, k) not in R})
    if miss:
        raise SystemExit(f"no measured step rate for {miss} in {CALS}: rerun analysis/data_saving/horizons.py first")
    total, half = 0.0, halved()
    for name, t, k, T, arm in jobs:
        p, h = write(name, t, k, T, arm, R[(t, k)], a.parent_nlo if t.endswith("_nlo") else a.parent_lo,
                     0.5 if (t, k) in half else 1.0)
        total += h
        print(f"{os.path.relpath(p, ROOT)}  {h:.1f} GPU-h")
    print(f"{len(jobs)} sweeps, {total:.1f} GPU-h estimated (8 trials each)")


if __name__ == "__main__":
    main()
