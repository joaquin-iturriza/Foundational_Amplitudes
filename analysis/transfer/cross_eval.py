"""Score trained runs on another recipe's pool: the MSE of log|M|^2 of each run's best checkpoint on the
test (or val) split of --recipe. The reference measure of the steering A/B (docs/results.tex,
sec:ladder) is the baseline's mixture test split: a run trained on the sigma-steered pool is scored
there, next to its baseline's own test loss; the steered test split is the second view. Each run is
rebuilt from its own config (tools/rebuild_run.py: its own train pool and stats, checked against its
frozen data_stats.json) with only the evaluated split's file replaced. The residual is in log|M|^2
(standardized residual times prepd_std; the target factors are exact and divide out of prediction and
truth alike), so runs on different pools share one unit. Prints one CROSS_EVAL JSON line per run.
Needs a GPU.
    python analysis/transfer/cross_eval.py <run_dir> [<run_dir> ...] --recipe recipes/<recipe>.yaml [--role test]
"""
import argparse, json, os, sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import datagen  # noqa: E402
import mg5_pipeline_final as mg  # noqa: E402
from rebuild_run import rebuild  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dirs", nargs="+")
ap.add_argument("--recipe", required=True, help="the recipe whose split the runs are scored on")
ap.add_argument("--role", default="test", choices=["val", "test"])
ap.add_argument("--seed", type=int, default=42, help="the recipe pools' seed (data.seed of the study)")
ap.add_argument("--dump", action="store_true",
                help="also save each event's residual with its pool row, sqrt(s) and raw |M|^2 to <run>/cross_eval_<role>.npz")
a = ap.parse_args()

rec = yaml.safe_load(open(a.recipe))
(spec,) = rec["processes"]
name, (lo, hi), n_eval = spec["name"], spec["sqrts"], int(spec[f"n_{a.role}"])


def eval_path():
    """The recipe's split, found under its own sampling policy (registered just for this lookup)."""
    mg.register_recipe_processes([{"name": name, "base": spec.get("base", name), "physics": spec.get("physics"),
                                   "sampling": None}], default_sampling=rec.get("sampling"))
    return datagen.ensure_dataset(name, lo, hi, n_eval, role=a.role, seed=a.seed, require_cache=True)


for run in a.run_dirs:
    path = eval_path()                     # before each rebuild: the run registers its own sampling
    exp, loader = rebuild(run, a.role, path, n_eval)
    with torch.no_grad():
        pred, truth, _ = exp._collect_predictions(loader)
    res = (np.asarray(pred, np.float64).reshape(-1) - np.asarray(truth, np.float64).reshape(-1)) * float(exp.prepd_std[0])
    if a.dump:
        perm = exp._role_perm[a.role][:len(res)]                  # position -> row of the eval pool file
        rows = np.asarray(np.load(path, mmap_mode="r")[perm])
        npart = (rows.shape[1] - 1) // 5
        mom = rows[:, :npart * 4].reshape(-1, npart, 4)
        tot = mom[:, 0] + mom[:, 1]
        np.savez_compressed(os.path.join(run, f"cross_eval_{a.role}.npz"), res=res, pool_row=perm,
                            sqrts=np.sqrt(np.maximum(tot[:, 0] ** 2 - (tot[:, 1:] ** 2).sum(1), 0)),
                            raw=rows[:, -1], momenta=mom, eval_pool=path)
    sq = np.sort(res ** 2)[::-1]
    print("CROSS_EVAL " + json.dumps(dict(run=os.path.abspath(run), recipe=os.path.abspath(a.recipe), eval_pool=path,
                                         role=a.role, n=int(len(sq)), mse=float(sq.mean()),
                                         share_top1pct=float(sq[:max(1, len(sq) // 100)].sum() / sq.sum()))),
          flush=True)
