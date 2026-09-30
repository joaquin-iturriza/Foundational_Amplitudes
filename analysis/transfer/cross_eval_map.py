"""Where a cross-evaluated run's error sits: the per-event residuals that cross_eval.py --dump saved
(<run>/cross_eval_<role>.npz) binned in sqrt(s) and cos theta (theta between the first beam and the
first final-state particle), as the MSE of log|M|^2 and the share of the squared error per bin, with the
event count per bin. Prints one JSON line. Numpy only; runs where the run lives.
    python analysis/transfer/cross_eval_map.py <run_dir> [--role test] [--bins 6]
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("--role", default="test"); ap.add_argument("--bins", type=int, default=6)
a = ap.parse_args()
Z = np.load(os.path.join(a.run_dir, f"cross_eval_{a.role}.npz"))
res, s, mom = Z["res"], Z["sqrts"], Z["momenta"]
p0, p2 = mom[:, 0, 1:], mom[:, 2, 1:]
cos = (p0 * p2).sum(1) / (np.linalg.norm(p0, axis=1) * np.linalg.norm(p2, axis=1))
sq = res ** 2
se = np.quantile(s, np.linspace(0, 1, a.bins + 1)); se[-1] *= 1 + 1e-9
ce = np.linspace(-1, 1 + 1e-9, a.bins + 1)
si, ci = np.digitize(s, se) - 1, np.digitize(cos, ce) - 1
mse = [[float(sq[(si == i) & (ci == j)].mean()) if ((si == i) & (ci == j)).any() else None for j in range(a.bins)] for i in range(a.bins)]
share = [[float(sq[(si == i) & (ci == j)].sum() / sq.sum()) for j in range(a.bins)] for i in range(a.bins)]
cnt = [[int(((si == i) & (ci == j)).sum()) for j in range(a.bins)] for i in range(a.bins)]
print(json.dumps(dict(run=a.run_dir, mse=float(sq.mean()), mean_res=float(res.mean()), sqrts_edges=se.tolist(),
                      cos_edges=ce.tolist(), mse_map=mse, share_map=share, count_map=cnt)))
