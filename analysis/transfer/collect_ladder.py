"""Collect the ladder's pretraining searches (tp3_ladder_r<n>) into one JSON, run on the site that holds them. Every
trial with a run directory: its searched HPs (from the run's config) and state. A finished trial carries its result
(val_loss_no_reg at the best checkpoint, best_step, the validation curve without the regularizer) and, from
plots_0/per_process_metrics.json and data_stats.json, each process's validation curve in MSE of log|M|^2 (the
standardized loss times that process's prepd_std^2); a running one the number of validations so far (a stopped one, its log silent for two hours, the same); a diverged one
the step (the log's "Training diverged" line). A diverged trial's value, its best checkpoint before the blow-up, is
not in its log (the printed loss carries the L2 term, which swamps it at large lambda, to four decimals): it comes
from tools/eval_best_val.py, kept in ladder_eval_best.json and read by ladder_pretrain.py.
Other pretraining searches by prefix (default tp3_ladder_r): python analysis/transfer/collect_ladder.py tp3_star_ tp3_pre64_
"""
import glob, json, os, re, sys, time
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import siteconf

HP = {"lr": ("training", "lr"), "lambda": ("training", "regularization_lambda"),
      "warmup": ("training", "cosanneal_warmup_frac"), "eta_min": ("training", "cosanneal_eta_min"),
      "ema_decay": ("training", "ema_decay"), "ema": (None, "ema")}


out = {}
for sdir in sorted(d for p in (sys.argv[1:] or ["tp3_ladder_r"]) for d in glob.glob(os.path.join(siteconf.RESULTS_DIR, p + "*"))):
    name = os.path.basename(sdir)
    res = {int(re.match(r"hp(\d+)_", os.path.basename(f)).group(1)): json.load(open(f))
           for f in glob.glob(os.path.join(sdir, "results", "hp*_t*.json"))}
    trials = []
    for run in sorted(glob.glob(os.path.join(siteconf.PROJECT_DIR, "runs", name, "trial_*"))):
        if not re.fullmatch(r"trial_\d+", os.path.basename(run)):
            continue
        hp = int(run.rsplit("_", 1)[1])
        if not os.path.exists(os.path.join(run, "config.yaml")):
            continue
        c = yaml.safe_load(open(os.path.join(run, "config.yaml")))
        t = {"hp": hp, **{k: (c.get(b) if a is None else c[a].get(b)) for k, (a, b) in HP.items()}}
        logs = sorted(glob.glob(os.path.join(run, "out_*.log")), key=os.path.getmtime)
        log = open(logs[-1]).read() if logs else ""
        m = re.search(r"Training diverged: .* at step (\d+)", log)
        if hp in res:
            t.update(state="done", **{k: v for k, v in res[hp].items() if k != "proc_val_losses"})
            pm, st = os.path.join(run, "plots_0", "per_process_metrics.json"), os.path.join(run, "data_stats.json")
            if os.path.exists(pm) and os.path.exists(st):
                m_, s_ = json.load(open(pm)), json.load(open(st))
                t["proc_curves"] = {p: [v * s_["prepd_std"][i] ** 2 for v in m_["proc_val_losses_no_reg"][p]]
                                    for i, p in enumerate(m_["dataset_order"])}
        elif m:
            t.update(state="diverged", step=int(m.group(1)))
        else:
            # no result and a log silent for two hours: the trial was stopped (cancelled, or its job lost)
            live = logs and time.time() - os.path.getmtime(logs[-1]) < 7200
            t.update(state="running" if live else "stopped", n_val=log.count("Val loss (combined)"))
        trials.append(t)
    out[name] = trials
json.dump(out, sys.stdout)
