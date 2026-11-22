"""Pooled test RMSE of log|M|^2 of a trained run on a recipe's test pools: Suprio's LLM-comparison metric (llm_compare),
sqrt of the mean over every scored event of (prediction - truth)^2 in log|M|^2, each process's first --eval-subsample
test events. The run is rebuilt on the memmap path with its best weights (EMA where it was on), from its own config with
only the recipe swapped:
  - its own recipe: the run's frozen statistics (data_stats.json), so this reproduces the run's own test stage;
  - another recipe (held-out processes): fine_tune.target_stats=parent_shared, each process standardized with the run's
    shared statistics (mu_N = a + b N and the one sigma under data.affine_multiplicity_std), no per-process map; the
    prediction is mu_N + sigma h, an absolute log|M|^2.
Writes --out (JSON): per process its MSE of log|M|^2, its MSE in the run's standardized units, events and particle
count; the pooled RMSE; the statistics used. Needs a GPU and the recipe's memmap cache (scripts/job_mmap_prebuild.sh).
    python tools/llm_score.py --run <run dir> --out <json> [--eval-subsample 10000] <recipe.yaml>
"""
import argparse, json, os, sys, tempfile

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
ap.add_argument("--eval-subsample", type=int, default=10000)
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
sse_z, n = np.zeros(len(names)), np.zeros(len(names), dtype=np.int64)
with torch.no_grad():
    for data in exp.test_loader:
        h, z, pid, _, _ = exp._forward_lloca(data)
        r2 = ((h.reshape(-1) - z.reshape(-1)).double() ** 2).cpu().numpy()
        p = pid.reshape(-1).cpu().numpy()
        np.add.at(sse_z, p, r2); np.add.at(n, p, 1)
assert (n > 0).all(), [nm for nm, k in zip(names, n) if k == 0]
mse_z = sse_z / n
mse_log = mse_z * sd ** 2
rec = {"run": run, "recipe": recipe, "target_stats": str(cfg.fine_tune.target_stats), "ema_applied": ema,
       "eval_subsample": a.eval_subsample, "affine_multiplicity": getattr(exp, "_affine_multiplicity", None),
       "prepd_std": sd.tolist(), "events": int(n.sum()),
       "pooled_rmse_logm2": float(np.sqrt((mse_log * n).sum() / n.sum())),
       "mean_mse_z": float(mse_z.mean()),
       "processes": {nm: {"mse_logm2": float(mse_log[i]), "mse_z": float(mse_z[i]), "n": int(n[i]),
                          "n_particles": int(exp._proc_npart[i])} for i, nm in enumerate(names)}}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(rec, open(a.out, "w"), indent=1)
print("LLM_SCORE " + json.dumps({k: rec[k] for k in ("recipe", "target_stats", "events", "pooled_rmse_logm2",
                                                       "mean_mse_z", "affine_multiplicity")}), flush=True)
