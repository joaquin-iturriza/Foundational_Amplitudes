"""Run one cell of a sweep config at FIXED hyperparameters instead of a DyHPO suggestion: the A/B protocol's
cheap shortcut (CLAUDE.md, A/B testing: the new setting at the baseline's best HPs). The run is built by
run_trial.build_command from the sweep config's fixed_params plus the given HPs, so it is the same run a
sweep trial would be; its result JSON (val_loss = the best checkpoint's val_loss_no_reg) goes to
<sweep_dir>/<name>/results/fixed.json.
    python sweep/run_fixed_hp.py --config sweep/<cfg>.yaml --name <cell name> --steps T key=value [key=value ...]
"""
import argparse, os, subprocess, sys
import yaml
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
import siteconf
from run_trial import build_command

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--steps", type=int, required=True); ap.add_argument("hp", nargs="*")
a = ap.parse_args()
cfg = siteconf.resolve(yaml.safe_load(open(a.config)))
cfg["sweep_name"] = a.name
def _val(v):
    """integers (seed) stay integers, numbers become floats, anything else (true/false) stays a string"""
    if v.lstrip("-").isdigit():
        return int(v)
    try:
        return float(v)
    except ValueError:
        return v
hp = {k: _val(v) for k, v in (x.split("=", 1) for x in a.hp)}
out = os.path.join(siteconf.SWEEP_DIR, a.name); os.makedirs(os.path.join(out, "results"), exist_ok=True)
run_dir = os.path.join(siteconf.PROJECT_DIR, "runs", a.name)
cmd = build_command(cfg, hp, run_dir, 0, os.path.join(out, "results", "fixed.json"), a.steps, increment_steps=a.steps)
print(" ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
