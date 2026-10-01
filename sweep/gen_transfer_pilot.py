"""Transfer study, pilot (chapter 1): pretrain on ee->uu alone, then the data-scaling curve of
two probes, a near one (ee->dd~) and a far one (uu~->gg), from scratch and fine-tuned; the other probes
(Z family, ladder) have scratch curves and, from the factor-off pretrain, fine-tunes too.

  pretrain   tp_pre_ee_uu              ee_uu, 100k train events, T_PRE steps
  scratch    tp2_scr_<probe>_d<k>      probe alone on the first D train events
  fine-tune  tp2_ft_<probe>_d<k>       the same, from the pretrain's best checkpoint
  tp3_       the same with data.target_propagators off, no Breit-Wigner factor either (--bw-off)
  tps_scr_   scratch on a sigma-steered pool (--steered)
The t-channel target factor is off (tp2_): on uu~->gg, uu~->Zg, ee->Za it set a floor near 1e-5
that switching it off removes, 10-290x lower at the same HPs (analysis/transfer/tchannel_ab.py);
the tp_ sweeps ran with it on and stay valid only for probes without a massless t/u-channel
exchange (ee->dd~, ee->tt~, the 2->3/2->4 and one-loop probes, where the factor is never applied).

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
  fine-tune           training.lr = the pretrain's best * sqrt(B/16384), fine_tune.lr_scale [0.1, 10]
                      (--lr-scale; the tp3_ fine-tunes search [1, 100]),
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
# the multiplicity chapter's fine-tune targets (the user's choice)
Z_FAMILY = ["uubar_Zg", "uubar_Zgg", "uubar_Zggg"]
# the ladder's probes, one or two per structure (EW t-channel, external photon, gluon exchange,
# masses, one loop). The one-loop pools hold 5e4 train events, so
# their grid stops at k = 9.
LADDER = ["ee_nnbar", "ee_Za", "ud_ud", "ee_ttbar", "ee_WW", "ee_dd_nlo", "ee_bb_nlo"]
K_MAX = {"ee_dd_nlo": 9, "ee_bb_nlo": 9}
# sigma-steered pools (tools/steer_pool.py; the user's call, 2026-09-30, after the ee->WW forward corner):
# the same probe on a pool built once from a reference model's sigma, scratch only, D <= 1e4, on the
# target of the probe's existing sweeps: ee->WW keeps the t-channel factor on, as its tp_scr sweeps, and
# its normalisation (median |t|, floor at the 1e-3 quantile) is pinned to the mixture pool's
# (analysis/transfer/ww_corner_ee_WW.json), since a pool steered into the forward corner would
# otherwise lower its own floor and change the target there. The DyHPO candidate seed is shared, so
# both arms search the same HP points. A/B order (CLAUDE.md): first the steered pool at each cell's
# baseline-best HPs (scripts/job_transfer_calib.sh fam=tps_scr), sweeps only where that loses.
_WW = json.load(open(os.path.join(ROOT, "analysis", "transfer", "ww_corner_ee_WW.json")))
STEERED = {"ee_WW": ("transfer_probe_ee_WW_steered.yaml",
                     {"data.target_propagator_tchannel": "true",
                      "data.target_propagator_tchannel_norm": f"[{_WW['median']!r},{_WW['floor']!r}]"})}
KS = range(2, 11)                                  # D = 10^(k/2)
T_PRE = 32000                                     # ee->dd~ at 1e5 events still improved at 16k
# Horizons from the calibration (analysis/transfer/calib_ee_ddbar.json, one fixed-HP run per D):
# a cell whose best checkpoint came early, while the lr was still high, gets 1.1x that step, so the
# cosine anneal lands where it had converged (the user's rule, 2026-09-30); a cell whose best
# checkpoint was at the end of its 8000 steps keeps 8000. k = 9, 10 were limited by the horizon and
# wait on the 32k/64k ladder. The horizons are measured on ee->dd~ and applied to every probe: the
# user's call (2026-09-30: "once we figure it out for one I don't expect that to change much for the
# other targets"), to be revisited where a probe's best trials all sit at the end of the horizon.
T_CELL = {2: 176, 3: 704, 4: 2816, 5: 4576, 6: 8000, 7: 8000, 8: 8000, 9: 16000, 10: 16000}
SETTLED = [2, 3, 4, 5, 6, 7, 8]
N_TRIALS, N_STARTUP = 8, 3
SEC_PER_STEP, OVERHEAD_MIN = 0.45, 15              # bs 16384 on a V100 (gen_solo16k_configs)
# The ladder's pretrainings (docs/results.tex sec:ladder, Protocol): one search per rung, run to convergence
# capped at 64k steps, factors off (tp3_), the lr window one decade either side of lr*(64k, high D).
T_LADDER = 64000

recs = json.load(open(os.path.join(ROOT, "analysis", "hpo_optima", "hpo_optima.json")))
row = collections.defaultdict(list)
for r in recs:
    if r["family"] == "scaling_p" and r["converged"] and r["hp_best"].get("training.lr") and r["n_train"] == 70000:
        row[r["t_steps"]].append(np.log(r["hp_best"]["training.lr"]["val"]))
ts = np.array(sorted(row)); lr_row = np.array([np.mean(row[t]) for t in ts])
# Past the high-D row's last horizon (~1.8e4) the centre follows CLAUDE.md rule 2's extrapolation of that
# row's decay (2e-3 at 3e4, 1.4e-3 at 5e4, 9e-4 at 1e5); np.interp alone would hold the last value
_EXTRAP = (np.log([3e4, 5e4, 1e5]), np.log([2e-3, 1.4e-3, 9e-4]))
lr_star = lambda t: float(np.exp(np.interp(np.log(t), np.r_[np.log(ts), _EXTRAP[0]], np.r_[lr_row, _EXTRAP[1]])
                                 if t > ts.max() else np.interp(np.log(t), np.log(ts), lr_row)))

FIXED = {
    "data.source": "recipes", "data.require_cache": "false", "data.eval_subsample": 10000,
    "data.phase_space_on_pool": "true",
    "data.preprocess_per_dataset": "true", "data.signedlog_quantile": 0.01, "data.seed": 42,
    "data.use_PIDs": "false", "data.spin_onehot": "true", "data.color_onehot": "true",
    "data.prop_is_massless": "true", "data.standardize_props": "true",
    "data.generation_onehot": "true", "data.generation_feature": "true",
    "data.mass_from_momenta": "false", "data.coupling_scalars": "true",
    "data.internal_mass_scalars": "true", "data.offshell_per_event": "true",
    "data.target_propagators": "true", "data.target_propagator_tchannel": "false",
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
                     "cpus_per_task": 4, "chain_width": 2,   # HTCondor DAG chained; SLURM: submit --chain
                     "time": "%02d:%02d:00" % divmod(max(30, int(T * SEC_PER_STEP / 60 * 1.5 + OVERHEAD_MIN)), 60)},
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
    ap.add_argument("--probes", nargs="*", help="only these probes (default: all)")
    ap.add_argument("--pretrain", action="store_true", help="also (re)write the pretrain config")
    ap.add_argument("--steered", action="store_true", help="write the steered-pool scratch configs (tps_scr_)")
    ap.add_argument("--lr-scale", default="0.1,10",
                    help="fine_tune.lr_scale window low,high (the tp2_ft searches' best sat at the top of 0.1-10 "
                         "in every cell, both probes: the tp3_ fine-tunes search 1-100)")
    ap.add_argument("--bw-off", action="store_true",
                    help="data.target_propagators off (tp3_): no Breit-Wigner factor either (the user's call, "
                         "2026-09-30, after the seeded A/B); the fixed-HP re-runs of the cells it touches")
    ap.add_argument("--ladder", nargs="*", type=int,
                    help="write the ladder's pretraining configs (tp3_ladder_r<r>) for these rungs, factors off")
    a = ap.parse_args()
    only = lambda ps: [p for p in ps if not a.probes or p in a.probes]
    out = []
    # the steered arm keeps its probe's current target (ee_WW: the t-channel factor on), so --bw-off never
    # rewrites it
    assert not (a.bw_off and a.steered), "--bw-off and --steered write different families; run them apart"
    pfx = "tp3" if a.bw_off else "tp2"
    if a.bw_off:
        FIXED["data.target_propagators"] = "false"
    if a.ladder:
        FIXED["data.target_propagators"] = "false"
        # tp3_ladder_r<n>: the tp3_pre_ladder_r<n> sweeps were centred on the ee_uu best instead and cancelled
        for r in a.ladder:
            out.append(write(f"tp3_ladder_r{r}", f"transfer_ladder_r{r}.yaml", T_LADDER, [lr_space(T_LADDER)] + COMMON_SPACE,
                             head=f"Transfer ladder, pretraining on rung {r} (cumulative), {T_LADDER} steps, factors off."))
        print("\n".join(os.path.relpath(p, ROOT) for p in out))
        return
    if a.steered:
        for p, (recipe, extra) in STEERED.items():
            for k in SETTLED:
                out.append(write(f"tps_scr_{p}_d{k}", recipe, T_CELL[k], [lr_space(T_CELL[k], k)] + COMMON_SPACE,
                                 dict({"data.train_subsample": int(round(10 ** (k / 2)))}, **extra),
                                 head=f"Transfer study, scratch on the sigma-steered pool: {p} on D = 10^{k / 2:g} events."))
    elif a.ft is None:
        if a.pretrain:
            out.append(write(f"{'tp' if pfx == 'tp2' else pfx}_pre_{PRE}", f"ref_solo_{PRE}.yaml", T_PRE, [lr_space(T_PRE)] + COMMON_SPACE,
                             head=f"Transfer pilot pretrain: {PRE} alone, 100k events, {T_PRE} steps."))
        for p in only(PROBES + Z_FAMILY + LADDER):
            for k in KS:
                if k > K_MAX.get(p, 10):
                    continue
                out.append(write(f"{pfx}_scr_{p}_d{k}", f"transfer_probe_{p}.yaml", T_CELL[k],
                                 [lr_space(T_CELL[k], k)] + COMMON_SPACE,
                                 {"data.train_subsample": int(round(10 ** (k / 2)))},
                                 head=f"Transfer pilot, scratch: {p} on D = 10^{k / 2:g} events."))
    else:
        assert a.lr, "--lr (the pretrain's best lr) is needed with --ft"
        ls_lo, ls_hi = (float(x) for x in a.lr_scale.split(","))
        ft_space = [{"name": "fine_tune.lr_scale", "type": "float_log", "low": ls_lo, "high": ls_hi},
                    {"name": "fine_tune.layer_decay", "type": "float_uniform", "low": 0.75, "high": 1.0}]
        # every probe is fine-tuned from the one pretrain (the plan, 2026-10-01); a probe's scratch arm must
        # share its target, so with --bw-off the probes whose target the factors change (the Z-window ones,
        # whose pools straddle the pole, and ee_WW, whose scratch arm kept the t-channel factor) need tp3_scr
        for p in only(PROBES + Z_FAMILY + LADDER):
            for k in KS:
                if k > K_MAX.get(p, 10):
                    continue
                out.append(write(f"{pfx}_ft_{p}_d{k}", f"transfer_probe_{p}.yaml", T_CELL[k], ft_space + COMMON_SPACE,
                                 {"data.train_subsample": int(round(10 ** (k / 2))), "training.lr": float(f"{a.lr * np.sqrt(batch(k) / 16384):.3g}"),
                                  "fine_tune.pretrained_path": a.ft, "fine_tune.target_stats": "own"},
                                 head=f"Transfer pilot, fine-tune from {PRE}: {p} on D = 10^{k / 2:g} events."))
    print("\n".join(os.path.relpath(p, ROOT) for p in out))
    print("lr* centres:", {T: f"{lr_star(T):.2g}" for T in sorted({T_PRE, *T_CELL.values()})})


if __name__ == "__main__":
    main()
