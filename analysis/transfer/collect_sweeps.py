"""Collect the transfer study's sweep trials into one JSON (run on the site that holds the sweeps).
Per trial: the result JSON (val/test loss at the best checkpoint, best_step, the validation curve),
the horizon, the run's prepd_std (data_stats.json, to convert to MSE of log|M|^2) and its lr, lr
from the run's config (a fine-tune also its lr_scale and layer_decay), and the other searched HPs (hps: lambda,
warm-up, eta_min, EMA and its decay), whether it was a random start-up trial and its place in the evaluation order. Sweeps are matched by prefix, e.g. tp_scr_ (a `_002` suffix is kept).
    python analysis/transfer/collect_sweeps.py tp_scr_ > out.json
"""
import glob, json, os, re, sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import siteconf

prefix = sys.argv[1]
out = {}
# a sweep's trials write to RESULTS_DIR (= SWEEP_DIR except on lxplus, EOS); a fixed-HP run (sweep/run_fixed_hp.py)
# writes results/fixed.json under SWEEP_DIR (AFS on lxplus), so both are read
names = sorted({os.path.basename(d) for root in {siteconf.RESULTS_DIR, siteconf.SWEEP_DIR}
                for d in glob.glob(os.path.join(root, prefix + "*"))})
for name in names:
    sdir = os.path.join(siteconf.RESULTS_DIR, name)
    trials, cand = [], None
    try:                                       # the DyHPO state: start-up candidates and evaluation order
        import pickle
        st_ = pickle.load(open(os.path.join(siteconf.SWEEP_DIR, name, "dyhpo_state.pkl"), "rb"))
        startup = {int(i) for i in st_.get("init_conf_indices", [])}
        order = {int(h): i for i, (h, _) in enumerate(st_.get("eval_order", []))}
    except Exception:
        startup, order = None, {}
    for rp in sorted(glob.glob(os.path.join(sdir, "results", "hp*_t*.json"))):
        m = re.match(r"hp(\d+)_t(\d+)_", os.path.basename(rp))
        hp, T = int(m.group(1)), int(m.group(2))
        r = json.load(open(rp))
        # the run dir that wrote this result: a cold-start rerun runs in trial_<hp>_r<k> (sweep/run_trial.py) beside the
        # failed attempt's trial_<hp>, so the dir is the one whose config names this result file
        base = os.path.join(siteconf.PROJECT_DIR, "runs", name, f"trial_{hp:04d}")
        run = base
        for d in [base] + sorted(glob.glob(base + "_r*")):
            try:
                rp_cfg = (yaml.safe_load(open(os.path.join(d, "config.yaml")))["training"] or {}).get("result_path")
            except (OSError, KeyError, TypeError, yaml.YAMLError):
                continue
            if rp_cfg and os.path.basename(str(rp_cfg)) == os.path.basename(rp):
                run = d
                break
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
                          **({"hps": hps} if hps else {}),
                          **({"startup": hp in startup, "order": order.get(hp)} if startup is not None else {}), **r))
    for fx in {os.path.join(root, name, "results", "fixed.json") for root in (siteconf.RESULTS_DIR, siteconf.SWEEP_DIR)}:
        if not os.path.exists(fx):
            continue
        # one fixed-HP run: its dir is runs/<name> itself (run_fixed_hp.py), hp = -1
        r = json.load(open(fx))
        run = os.path.join(siteconf.PROJECT_DIR, "runs", name)
        std = lr = hps = ft = T = None
        if os.path.exists(os.path.join(run, "data_stats.json")):
            std = json.load(open(os.path.join(run, "data_stats.json")))["prepd_std"][0]
        if os.path.exists(os.path.join(run, "config.yaml")):
            c = yaml.safe_load(open(os.path.join(run, "config.yaml")))
            lr, T = c["training"]["lr"], c["training"].get("iterations")
            hps = {"lambda": c["training"].get("regularization_lambda"), "warmup": c["training"].get("cosanneal_warmup_frac"),
                   "eta_min": c["training"].get("cosanneal_eta_min"), "ema": c.get("ema"),
                   "ema_decay": c["training"].get("ema_decay")}
            if (c.get("fine_tune") or {}).get("pretrained_path"):
                ft = {k: c["fine_tune"].get(k) for k in ("lr_scale", "layer_decay")}
        trials.append(dict(hp=-1, T=T, fixed=True, prepd_std=std, lr=lr, **({"fine_tune": ft} if ft else {}),
                           **({"hps": hps} if hps else {}), **r))
        break
    if trials or os.path.isdir(sdir):
        out[name] = trials
json.dump(out, sys.stdout)
