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
             training.lr = c, fine_tune.layer_decay [0.75, 1]: --ft-space llrd, the one-loop default) or scratch's window
             (--ft-space scratch: layer_decay 1, lr_scale [1/3, 3], the tree default); the target's own amplitude stats, the parent's
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
import re
import sys
import zlib

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
TARGETS = ["ee_uu", "epem_ttx", "ee_uu_nlo", "ee_ttbar_nlo"]
SECOND = ["bbx_zz_nlo", "bbx_zz"]       # D413, D417: the draft's second example (b b~ > Z Z at LO and one-loop V/B), opt-in by --targets
CALS = [os.path.join(ROOT, "analysis", "data_saving", "data", f) for f in
        ("horizons_cal_f22_cal_f22_lrhalf.json", "horizons_cal2_ladder.json", "horizons_cal_vb_ladder.json",
         "horizons_cal_lo_ladder.json")]
VB = {"ee_ttbar_nlo", "bbx_zz_nlo"}                   # trained on V/B (D375), physics factors off (job_data_saving.sh)
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
    return [r for f in CALS if os.path.exists(f) for r in json.load(open(f)) if r["target"] in TARGETS + SECOND]


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


def write(name, tgt, k, T, arm, rate, parent, lr_factor=1.0, offshell="parent", ft_space="llrd", tmpl2=False,
          wsd=False, batch_search=False, warmup_high=None):
    c = lr_centre(T, k) * lr_factor
    space = template_space()
    if warmup_high:
        # step 6 warmup test: the warmup range opened upward, the peak lr reached later in the run
        space = [dict(s, high=warmup_high) if s["name"] == "training.cosanneal_warmup_frac" else s for s in space]
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
                      "fine_tune.target_stats": "own", "fine_tune.offshell_stats": offshell})
        if ft_space == "scratch":
            # step 3 (2026-11-12): the scratch arm's space, the parent's weights the only difference: one param group
            # (layer_decay 1) and lr = c x lr_scale over scratch's [c/3, 3c]
            fixed["fine_tune.layer_decay"] = 1.0
            space = [{"name": "fine_tune.lr_scale", "type": "float_log", "low": 0.333, "high": 3.0}] + space
        else:
            # tmpl2 (hp_ranges_readout.py, 2026-11-12): the stage-2 bests sat at the low edge of both in 9 of 30
            # searches at D < 1e4, so both ranges open downward
            # wide (no batch search): also opened upward, where the stage-2 good fine-tune trials sat (place 0.955 of
            # [0.1, 10], hp_ranges_readout.json; design FIX 2026-11-12)
            space = [{"name": "fine_tune.lr_scale", "type": "float_log", "low": 0.03 if tmpl2 else 0.1,
                      "high": 30.0 if tmpl2 and not batch_search else 10.0},
                     {"name": "fine_tune.layer_decay", "type": "float_uniform", "low": 0.5 if tmpl2 else 0.75,
                      "high": 1.0}] + space
    if tmpl2 or wsd:
        # D423 (Joaquin 2026-11-12): WSD as the schedule for both arms, its cooldown fixed at the last eighth as the
        # WSD parent's recipe (results.tex par. v4 WSD, FA-ccin2p3-388dcd45); warmup and floor stay searched
        fixed.update({"training.scheduler": "WSD", "training.wsd_cooldown_frac": 0.125})
    if tmpl2 and batch_search:
        # batch size, fixed at D/2 in stage 2, searched: no searched HP moves the val/train separation within its
        # range (|rho| <= 0.10 but lr's -0.30), while it is set by D, and batch size and horizon are what follow D
        space = [{"name": "training.batchsize", "type": "int_log", "low": max(4, round(d_of(k) / 64)),
                  "high": int(d_of(k) // 2)}] + space
    hours = T * rate / 3600 + OVERHEAD_H
    # 11G: dsave trials peak <= 1.8 GB RSS; 32G became 11 CPUs on lxplus, which its V100 nodes never offer (E6895)
    cfg = {"cluster": {"scheduler": "slurm", "auto_submit": False, "request_gpus": 1, "mem": "11G",
                       "cpus_per_task": 4,
                       # the estimate itself: sweep/drive.py applies the one margin (1.5x + 6 min)
                       "time": "%02d:%02d:00" % divmod(max(1, int(round(hours * 60))), 60)},
           "paths": None, "sweep_name": name, "base_config": base, "n_trials": N_TRIALS,
           "dyhpo": {"surrogate": "gp", "n_candidates": 2000, "objective_scale": "auto",
                     "seed": zlib.crc32(re.sub(r"^dsave_s\d+[a-z]*_", "dsave_s2_", name).encode()) % 2 ** 31, "n_startup": N_STARTUP, "total_budget": 10000,
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
    ap.add_argument("--stage", default="s2", help="sweep-name stage tag: s2, s3 (the fine-tunes from the D400 parents)")
    ap.add_argument("--offshell-nlo", default="own", choices=["parent", "own"],
                    help="fine_tune.offshell_stats of the one-loop fine-tunes: own, as results.tex adopts for the one-loop "
                         "fine-tunes from the s3 searches on (diag_readout.py @5555a0f0); parent reproduces s2/s4")
    ap.add_argument("--ft-space", choices=["llrd", "scratch"],
                    help="fine-tune search space: llrd (lr_scale [0.1,10], layer_decay [0.75,1]) or scratch (layer_decay 1, "
                         "lr_scale [1/3,3]: scratch's lr window); default scratch for tree targets (step-3 test FIXED on "
                         "t t~ tree, s4_readout.py), llrd for one-loop (IMPROVED, NOT RESOLVED: no default change)")
    ap.add_argument("--lr-factor", type=float, default=1.0,
                    help="multiplies the lr centre c of every config (both arms); curve_quality.py: the current c gives "
                         "spiking, jittering val curves in every t t~ / u u~ search at D>=316")
    ap.add_argument("--tmpl2", action="store_true",
                    help="the revised template (hp_ranges_readout.py, D423): WSD (cooldown 1/8) for both arms, batch D/2 (searched only with --batch-search), fine-tune "
                         "lr_scale [0.03, 10] and layer_decay [0.5, 1]")
    ap.add_argument("--wsd", action="store_true", help="the stage-2 space under WSD (cooldown 1/8) only: the matched "
                    "control of --tmpl2's batch and ranges")
    ap.add_argument("--batch-search", action="store_true",
                    help="--tmpl2 with the batch size searched in [D/64, D/2]: forbidden by D426 (searching batch beside lr "
                         "confounds the two); kept only to reproduce the 4 tmpl2 configs made before it")
    ap.add_argument("--no-batch-search", action="store_true", help="the default since D426; kept so old commands run")
    ap.add_argument("--warmup-high", type=float, help="upper edge of the searched training.cosanneal_warmup_frac "
                    "(template 0.2): the step-6 fine-tune warmup test")
    ap.add_argument("--ks", type=int, nargs="*", help="only these D exponents k (D = 10^(k/2))")
    a = ap.parse_args()
    R, hz = rates(), horizons()
    jobs = []
    for t in a.targets:
        for k in range(2, 10 if t.endswith("_nlo") else 11):
            if a.ks and k not in a.ks:
                continue
            T = hz.get(t, {}).get(k)
            if T is None:
                raise SystemExit(f"no T(D) for {t} k={k}: rerun analysis/data_saving/horizons.py --ladder")
            for arm in a.arms:
                jobs.append((f"dsave_{a.stage}_{arm}_{t}_d{k}_t{T}", t, k, T, arm))
    miss = sorted({(t, k) for _, t, k, _, _ in jobs if (t, k) not in R})
    if miss:
        raise SystemExit(f"no measured step rate for {miss} in {CALS}: rerun analysis/data_saving/horizons.py first")
    total, half = 0.0, halved()
    for name, t, k, T, arm in jobs:
        p, h = write(name, t, k, T, arm, R[(t, k)], a.parent_nlo if t.endswith("_nlo") else a.parent_lo,
                     (0.5 if (t, k) in half else 1.0) * a.lr_factor, a.offshell_nlo if t.endswith("_nlo") else "parent",
                     a.ft_space or ("llrd" if t.endswith("_nlo") else "scratch"), a.tmpl2, a.wsd, a.batch_search,
                     a.warmup_high)
        total += h
        print(f"{os.path.relpath(p, ROOT)}  {h:.1f} GPU-h")
    print(f"{len(jobs)} sweeps, {total:.1f} GPU-h estimated (8 trials each)")


if __name__ == "__main__":
    main()
