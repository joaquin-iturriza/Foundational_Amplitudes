"""Collect the transfer study's fixed-HP runs (scripts/job_transfer_calib.sh: tp_calib_* before the
t-channel factor was switched off, tp2_calib_* after) into one JSON, run on the site that holds them.
Per run: the result JSON (val/test loss at the best checkpoint, best_step, the validation curve),
the run's prepd_std (data_stats.json, for MSE of log|M|^2), and from its config.yaml the HPs and the
target flags actually used (target_propagators, target_propagator_tchannel), so the factor-on and
factor-off arms are told apart by what ran, not by the name. Writes analysis/transfer/long_runs.json
and tchannel_ab.json once merged across sites (see the plotting scripts).
    python analysis/transfer/collect_calib.py > out.json
"""
import glob, json, os, sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import siteconf

out = {}
for f in sorted(glob.glob(os.path.join(siteconf.RESULTS_DIR, "tp*_calib_*", "results", "fixed.json"))):
    name = f.split(os.sep)[-3]
    run = os.path.join(siteconf.PROJECT_DIR, "runs", name)
    try:
        r = json.load(open(f))
        c = yaml.safe_load(open(os.path.join(run, "config.yaml")))
        std = json.load(open(os.path.join(run, "data_stats.json")))["prepd_std"][0]
    except (OSError, KeyError, ValueError):
        continue
    t, d = c["training"], c["data"]
    out[name] = dict(val=r["val_loss"], best_step=r.get("best_step"), hours=r["traintime_hours"],
                     std=std, curve=r.get("val_curve"), every=r.get("validate_every"),
                     lr=t["lr"], warmup=t["cosanneal_warmup_frac"], lam=t["regularization_lambda"],
                     eta_min=t["cosanneal_eta_min"], ema=c.get("ema"),
                     target_propagators=d.get("target_propagators"),
                     tchannel=d.get("target_propagator_tchannel"), site=siteconf.SITE)
json.dump(out, sys.stdout)
