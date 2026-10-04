"""Collect the transfer study's sweep trials into one JSON (run on the site that holds the sweeps).
Per trial: the result JSON (val/test loss at the best checkpoint, best_step, the validation curve),
the horizon, the run's prepd_std (data_stats.json, to convert to MSE of log|M|^2) and its lr, lr
from the run's config (a fine-tune also its lr_scale and layer_decay), and the other searched HPs (hps: lambda,
warm-up, eta_min, EMA and its decay). Sweeps are matched by prefix, e.g. tp_scr_ (a `_002` suffix is kept).
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
    trials, cand = [], None
    for rp in sorted(glob.glob(os.path.join(sdir, "results", "hp*_t*.json"))):
        m = re.match(r"hp(\d+)_t(\d+)_", os.path.basename(rp))
        hp, T = int(m.group(1)), int(m.group(2))
        r = json.load(open(rp))
        run = os.path.join(siteconf.PROJECT_DIR, "runs", name, f"trial_{hp:04d}")
        std = lr = ft = hps = None
        if os.path.exists(os.path.join(run, "data_stats.json")):
            std = json.load(open(os.path.join(run, "data_stats.json")))["prepd_std"][0]
        if os.path.exists(os.path.join(run, "config.yaml")):
            c = yaml.safe_load(open(os.path.join(run, "config.yaml")))
            lr = c["training"]["lr"]
            hps = {"lambda": c["training"].get("regularization_lambda"), "warmup": c["training"].get("cosanneal_warmup_frac"),
                   "eta_min": c["training"].get("cosanneal_eta_min"), "ema": c.get("ema"),
                   "ema_decay": c["training"].get("ema_decay")}
            if (c.get("fine_tune") or {}).get("pretrained_path"):
                ft = {k: c["fine_tune"].get(k) for k in ("lr_scale", "layer_decay")}
        if hps is None or lr is None:          # run dir moved off this site (EOS cleanup): the sweep's own candidates
            if cand is None:
                try:
                    import pickle
                    cand = pickle.load(open(os.path.join(siteconf.SWEEP_DIR, name, "dyhpo_state.pkl"), "rb"))["candidates_raw"]
                except Exception:
                    cand = {}
            c_ = cand.get(hp) if isinstance(cand, dict) else cand[hp] if hp < len(cand) else None
            if c_:
                lr = lr if lr is not None else c_.get("training.lr")
                hps = {"lambda": c_.get("training.regularization_lambda"), "warmup": c_.get("training.cosanneal_warmup_frac"),
                       "eta_min": c_.get("training.cosanneal_eta_min"), "ema": c_.get("ema"),
                       "ema_decay": c_.get("training.ema_decay")}
        trials.append(dict(hp=hp, T=T, prepd_std=std, lr=lr, **({"fine_tune": ft} if ft else {}),
                          **({"hps": hps} if hps else {}), **r))
    out[name] = trials
json.dump(out, sys.stdout)
