"""Pooled test RMSE of log|M|^2 of a trained run on a recipe's test pools: Suprio's LLM-comparison metric (llm_compare),
sqrt of the mean over every scored event of (prediction - truth)^2 in log|M|^2, each process's first --eval-subsample
test events. The run is rebuilt on the memmap path with its best weights (EMA where it was on), from its own config with
only the recipe swapped:
  - its own recipe: the run's frozen statistics (data_stats.json), so this reproduces the run's own test stage;
  - another recipe (held-out processes): fine_tune.target_stats=parent_shared, each process standardized with the run's
    shared statistics (mu_N = a + b N and the one sigma under data.affine_multiplicity_std), no per-process map; the
    prediction is mu_N + sigma h, an absolute log|M|^2.
Writes --out (JSON): per process its MSE of log|M|^2, its MSE in the run's standardized units, events and particle
count, its exact split MSE = bias^2 + scale + floor (bias = mean(pred - true); floor = var(true)(1 - rho^2), left by the
best affine map; scale the rest), rho = corr(pred, true) and the true mean and std, all in absolute log|M|^2 (target
factor divided out); the pooled RMSE; the statistics used. Needs a GPU and the recipe's memmap cache (scripts/job_mmap_prebuild.sh).
    python tools/llm_score.py --run <run dir> --out <json> [--eval-subsample N] <recipe.yaml>
"""
import argparse, json, os, shutil, sys, tempfile

import numpy as np
import torch
from omegaconf import open_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import siteconf  # noqa: E402
from base_experiment import clear_start_only  # noqa: E402
from rebuild_run import _build, load_best_weights, load_cfg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--eval-subsample", type=int, default=None, help="test events per process (default: the run's own data.eval_subsample)")
ap.add_argument("--events-out", default=None, help="CSV of process, true and predicted log|M|^2 per test event (D567)")
ap.add_argument("--events-max", type=int, default=5000, help="events per process written to --events-out")
ap.add_argument("recipe")
a = ap.parse_args()

run = os.path.abspath(a.run)
recipe = os.path.abspath(siteconf._expand_str(a.recipe))
cfg = load_cfg(run)
own = os.path.abspath(siteconf._expand_str(str(cfg.data.processes_file))) == recipe
ckpt = os.path.join(run, "models", "model_run0_best.pt")
with open_dict(cfg):
    cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
    cfg.use_mlflow = False; cfg.count_flops = False; cfg.ema = False; cfg.log_module_names = False
    cfg.warm_start_idx = None
    cfg.run_dir = tempfile.mkdtemp(prefix="llmscore_", dir=os.environ["SCRATCH"])
    cfg.data.processes_file = recipe
    if a.eval_subsample is not None:
        cfg.data.eval_subsample = a.eval_subsample
    cfg.fine_tune.pretrained_path = ckpt
    cfg.fine_tune.target_stats = "parent" if own else "parent_shared"
    clear_start_only(cfg.fine_tune)
exp = _build(cfg, None, None, None)
ema = load_best_weights(exp.model, run)
exp.model.to(exp.device, dtype=exp.dtype).eval()
exp.ema, exp.train_sampler = None, None
exp._init_regularization()

names = [str(n) for n in exp.cfg.data.dataset]
sd = np.asarray(exp.prepd_std, dtype=np.float64)
sd = np.full(len(names), sd[0]) if sd.size == 1 else sd
mu = np.asarray(exp.prepd_mean, dtype=np.float64).ravel()
mu = np.full(len(names), mu[0]) if mu.size == 1 else mu
# the per-event target factor (target_physics_factor, propagators: the target is |M|^2 F) in the test loader's order,
# divided out so pred and true are absolute log|M|^2 (as the test stage's raw arrays, check_score_vs_preds.py)
F = getattr(exp, "all_amp_factor", None)
assert F is not None or not bool(cfg.data.get("target_physics_factor", False)), "target factor on but no all_amp_factor"
logF = np.log(np.asarray(F, dtype=np.float64)[exp._split_indices["test"]]).ravel() if F is not None else None
sse_z, n = np.zeros(len(names)), np.zeros(len(names), dtype=np.int64)
S = np.zeros((len(names), 5))
EV = []   # per process sums of pred, true, pred^2, true^2, pred true (absolute log|M|^2)
k = 0
with torch.no_grad():
    for data in exp.test_loader:
        h, z, pid, _, _ = exp._forward_lloca(data)
        hd, zd = h.reshape(-1).double().cpu().numpy(), z.reshape(-1).double().cpu().numpy()
        p = pid.reshape(-1).cpu().numpy()
        np.add.at(sse_z, p, (hd - zd) ** 2); np.add.at(n, p, 1)
        lf = logF[k:k + p.size] if logF is not None else 0.0
        q, t = mu[p] + sd[p] * hd - lf, mu[p] + sd[p] * zd - lf
        np.add.at(S, p, np.stack([q, t, q * q, t * t, q * t], 1))
        if a.events_out:
            EV.append(np.stack([p, t, q, np.broadcast_to(lf, p.shape)], 1))   # + the target factor log F (0 without)
        k += p.size
assert logF is None or k == logF.size, (k, None if logF is None else logF.size)
assert (n > 0).all(), [nm for nm, k in zip(names, n) if k == 0]
mse_z = sse_z / n
mse_log = mse_z * sd ** 2
# MSE = bias^2 + scale + floor, exactly: bias = mean(q - t); floor = var(t)(1 - rho^2), what the best affine map of q
# leaves (hp58's 1 - rho^2, docs/results.tex 'Its held-out split'); scale = var(q - t) - floor, what the best rescale removes
mq, mt = S[:, 0] / n, S[:, 1] / n
vq, vt, cov = S[:, 2] / n - mq ** 2, S[:, 3] / n - mt ** 2, S[:, 4] / n - mq * mt
bias_log = mq - mt
rho = cov / np.sqrt(np.clip(vq * vt, 1e-300, None))
floor_log = vt * (1 - rho ** 2)
scale_log = mse_log - bias_log ** 2 - floor_log
assert np.allclose(vq + vt - 2 * cov + bias_log ** 2, mse_log, rtol=1e-6, atol=1e-9), "absolute-log sums disagree with the MSE"
rec = {"run": run, "recipe": recipe, "target_stats": str(cfg.fine_tune.target_stats), "ema_applied": ema,
       "eval_subsample": cfg.data.eval_subsample, "amp_trafos": [list(map(str, t)) for t in (exp._amp_trafos_pp or [exp.cfg.data.amp_trafos])], "affine_multiplicity": getattr(exp, "_affine_multiplicity", None),
       "prepd_std": sd.tolist(), "events": int(n.sum()),
       "pooled_rmse_logm2": float(np.sqrt((mse_log * n).sum() / n.sum())),
       "mean_mse_z": float(mse_z.mean()),
       "processes": {nm: {"mse_logm2": float(mse_log[i]), "mse_z": float(mse_z[i]), "n": int(n[i]),
                          "n_particles": int(exp._proc_npart[i]), "bias_logm2": float(bias_log[i]),
                          "scale_logm2": float(scale_log[i]), "floor_logm2": float(floor_log[i]), "rho": float(rho[i]),
                          "true_mean_logm2": float(mt[i]), "true_std_logm2": float(np.sqrt(max(vt[i], 0.0)))}
                     for i, nm in enumerate(names)}}
if a.events_out:  # the first --events-max test events of each process, in loader order
    ev = np.concatenate(EV)
    keep = np.concatenate([np.flatnonzero(ev[:, 0] == i)[: a.events_max] for i in range(len(names))])
    with open(a.events_out, "w") as f:
        f.write("process,true_logm2,pred_logm2,log_factor\n")
        f.writelines("%s,%.6g,%.6g,%.6g\n" % (names[int(r[0])], r[1], r[2], r[3]) for r in ev[keep])
shutil.rmtree(cfg.run_dir, ignore_errors=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(rec, open(a.out, "w"), indent=1)
print("LLM_SCORE " + json.dumps({k: rec[k] for k in ("recipe", "target_stats", "events", "pooled_rmse_logm2",
                                                       "mean_mse_z", "affine_multiplicity")}), flush=True)
