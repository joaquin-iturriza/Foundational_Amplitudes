"""Collect the transfer study's sweep trials into one JSON (run on the site that holds the sweeps).
Per trial: the result JSON (val/test loss at the best checkpoint, best_step, the validation curve),
the horizon, the run's prepd_std (data_stats.json, to convert to MSE of log|M|^2) and its lr, lr
from the run's config. Sweeps are matched by prefix, e.g. tp_scr_ (a `_002` suffix is kept).
    python analysis/transfer/collect_sweeps.py tp_scr_ > out.json
"""
import glob, json, os, re, sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import siteconf

prefix = sys.argv[1]
out = {}
for sdir in sorted(glob.glob(os.path.join(siteconf.RESULTS_DIR, prefix + "*"))):   # = SWEEP_DIR except on lxplus (EOS)
    name = os.path.basename(sdir)
    trials = []
    for rp in sorted(glob.glob(os.path.join(sdir, "results", "hp*_t*.json"))):
        m = re.match(r"hp(\d+)_t(\d+)_", os.path.basename(rp))
        hp, T = int(m.group(1)), int(m.group(2))
        r = json.load(open(rp))
        run = os.path.join(siteconf.PROJECT_DIR, "runs", name, f"trial_{hp:04d}")
        std = lr = None
        if os.path.exists(os.path.join(run, "data_stats.json")):
            std = json.load(open(os.path.join(run, "data_stats.json")))["prepd_std"][0]
        if os.path.exists(os.path.join(run, "config.yaml")):
            lr = yaml.safe_load(open(os.path.join(run, "config.yaml")))["training"]["lr"]
        trials.append(dict(hp=hp, T=T, prepd_std=std, lr=lr, **r))
    out[name] = trials
json.dump(out, sys.stdout)
