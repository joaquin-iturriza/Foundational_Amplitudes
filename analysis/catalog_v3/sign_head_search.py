"""The sign head's search on catalog_v3 (sweep/sweep_config_ccin2p3_sign_head_search.yaml; plan entry v3_prep_explore;
docs/results.tex par:v3_prep_signed). The question: once its own knobs and the learning rate are searched with Muon, does
the sign head carry the one-loop sign to held-out processes? A single-fidelity DyHPO, 20 trials at the short
pretraining's 4264 steps, every trial the more-signed-pools arm (baseline target, one shared standardization,
recipes/catalog_v3_train_signext.yaml, 1492 pools, seed 42) with MuonAdamW at decoupled weight decay 0.0217; searched the
lr, the warmup fraction, the pair sign head (width 0 = off, hidden, tanh bound) and the near-zero weighting (weight 1 =
off, band). Objective val_total: the magnitude MSE plus the sign cross-entropy on training-process validation events,
geometric mean over processes, at the best val_loss_no_reg checkpoint.
  collect (on the site): every trial's result file (val_loss = best-checkpoint val_loss_no_reg, objective_value =
           val_total there, the curve), its run config (the HPs it ran), the failed trials' logs; and the two reference
           runs' result files (the Muon baseline baseline_more_signed_muon, the AdamW arm baseline_more_signed):
    site run --item v3_prep_explore ccin2p3 FA -- python analysis/catalog_v3/sign_head_search.py collect \
        > analysis/catalog_v3/data/sign_head_search_trials.json
  main: every trial's objective and magnitude loss with its settings; blow-ups (a validation over BLOWUP_FACTOR times its
        running minimum, pair_head_blowup.py's rule) listed with their step and kept in every number; the best trial
        with both heads off (pair width 0 and near weight 1) against the best with either on, inside the search; both
        against the Muon baseline's own val_total and val_loss_no_reg (its lr and weight decay come from the Muon search
        on the dimensionless target, so the in-search comparison is the fair one). If the best trial's zero-shot reads
        are home (data/short_reads/sign_search_best/, arm sign_search_best of scripts/job_catalog_v3_short.sh, mode
        eval), its held-out wrong-sign share and zero-shot MSE per probe against the Muon baseline's and the AdamW
        arm's, as prep_signed.py reads them.
    python3 analysis/catalog_v3/sign_head_search.py   -> data/sign_head_search.json, figs/sign_head_search.{png,pdf}
Figure (1x2): (a) every trial's objective against its lr, heads off and on marked apart, blow-ups hollow, the Muon
baseline's line; (b) the magnitude loss against the objective, same markers.
"""
import glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
SWEEP = "sign_head_search"
HPS = ("training.lr", "training.cosanneal_warmup_frac", "model.sign_pair_dim", "model.sign_pair_hidden",
       "model.sign_pair_bound", "training.sign_near_weight", "training.sign_near_factor", "training.optimizer",
       "training.muon_weight_decay", "seed")
REFS = {"muon_baseline": "runs/catalog_v3_short/baseline_more_signed_muon",
        "adamw_more_signed": "runs/catalog_v3_short/baseline_more_signed"}
DATA = os.path.join(HERE, "data", "sign_head_search_trials.json")


def _get(c, k):
    for p in k.split("."):
        if not isinstance(c, dict) or p not in c:
            return None
        c = c[p]
    return c


def collect():
    import yaml
    small = lambda r: {k: v for k, v in r.items() if not isinstance(v, dict) or k.startswith("val_total")}
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "sweeps", SWEEP, "results", "hp*_t*.json"))):
        hp = int(os.path.basename(f)[2:6])
        r = json.load(open(f))
        d = os.path.join(ROOT, "runs", SWEEP, f"trial_{hp:04d}")
        c = yaml.safe_load(open(os.path.join(d, "config.yaml")))
        rj = os.path.join(d, "result.json")
        rows.append({"hp_idx": hp, "result_file": small(r), "hps": {k: _get(c, k) for k in HPS},
                     "run_result": small(json.load(open(rj))) if os.path.exists(rj) else None,
                     "has_best_model": any(os.path.exists(os.path.join(d, "models", "model_run0_best.pt" + z))
                                           for z in ("", ".gz"))})
    failed = [os.path.basename(e) for e in glob.glob(os.path.join(ROOT, "sweeps", SWEEP, "error", "*.err"))
              if "Trial FAILED" in open(e, errors="replace").read()]
    refs = {}
    for k, d in REFS.items():
        rj = os.path.join(ROOT, d, "result.json")
        refs[k] = small(json.load(open(rj))) if os.path.exists(rj) else None
    json.dump({"sweep": SWEEP, "trials": rows, "failed_logs": failed, "refs": refs,
               "n_jobs": len(glob.glob(os.path.join(ROOT, "sweeps", SWEEP, "jobs", "trial_*.sh")))}, sys.stdout)


if __name__ == "__main__":
    if sys.argv[1:] == ["collect"]:
        collect()
