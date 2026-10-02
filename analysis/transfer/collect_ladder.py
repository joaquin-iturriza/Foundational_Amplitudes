"""Collect the ladder's pretraining searches (tp3_ladder_r<n>) into one JSON, run on the site that holds them. Every
trial with a run directory: its searched HPs (from the run's config) and state. A finished trial carries its result
(val_loss_no_reg at the best checkpoint, best_step, the validation curve without the regularizer) and, from
plots_0/per_process_metrics.json and data_stats.json, each process's validation curve in MSE of log|M|^2 (the
standardized loss times that process's prepd_std^2); a running one the number of validations so far; a diverged one
the step (the log's "Training diverged" line) and, read back from its log, the same result up to the blow-up: its best
checkpoint is still its value (CLAUDE.md, reported values). The log prints each validation's regularized aggregate
v = GM_p(m_p) + r and each process's m_p + r (r the L2 term, one number), to 4 significant figures; r is the root
of GM_p(l_p - r) = v - r, so val_loss_no_reg = v - r and m_p = l_p - r.
    python analysis/transfer/collect_ladder.py > out.json
"""
import glob, json, os, re, sys
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import siteconf

HP = {"lr": ("training", "lr"), "lambda": ("training", "regularization_lambda"),
      "warmup": ("training", "cosanneal_warmup_frac"), "eta_min": ("training", "cosanneal_eta_min"),
      "ema_decay": ("training", "ema_decay"), "ema": (None, "ema")}


def from_log(log):
    """The validation curve without the regularizer, each process's, and the validation interval, from a log."""
    every = int(re.search(r"validating every (\d+) steps", log).group(1))
    curve, procs = [], {}
    for v, rest in re.findall(r"Val loss \(combined\): ([0-9.eE+-]+) \| (.*)", log):
        l = {n: float(x) for n, x in (kv.split("=") for kv in rest.split(", "))}
        v, L = float(v), np.array(list(l.values()))
        f = lambda r: np.exp(np.mean(np.log(np.clip(L - r, 1e-300, None)))) - (v - r)
        lo, hi = 0.0, L.min() * (1 - 1e-12)
        if f(lo) * f(hi) < 0:
            for _ in range(200):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
            r = (lo + hi) / 2
        else:                                   # rounding left no sign change: the closer end
            r = lo if abs(f(lo)) < abs(f(hi)) else hi
        curve.append(v - r)
        for n, x in l.items():
            procs.setdefault(n, []).append(x - r)
    return curve, procs, every


out = {}
for sdir in sorted(glob.glob(os.path.join(siteconf.RESULTS_DIR, "tp3_ladder_r*"))):
    name = os.path.basename(sdir)
    res = {int(re.match(r"hp(\d+)_", os.path.basename(f)).group(1)): json.load(open(f))
           for f in glob.glob(os.path.join(sdir, "results", "hp*_t*.json"))}
    trials = []
    for run in sorted(glob.glob(os.path.join(siteconf.PROJECT_DIR, "runs", name, "trial_*"))):
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
            curve, procs, every = from_log(log)
            if curve:
                i = int(np.argmin(curve))
                t.update(val_loss=curve[i], best_step=(i + 1) * every, validate_every=every, val_curve=curve,
                         from_log=True)
                st = os.path.join(run, "data_stats.json")
                if os.path.exists(st):
                    s_ = json.load(open(st))
                    # the log names the processes in dataset order, the order of prepd_std
                    assert len(procs) == len(s_["prepd_std"]), (run, list(procs))
                    t["proc_curves"] = {p: [x * sd ** 2 for x in procs[p]] for p, sd in zip(procs, s_["prepd_std"])}
        else:
            t.update(state="running", n_val=log.count("Val loss (combined)"))
        trials.append(t)
    out[name] = trials
json.dump(out, sys.stdout)
