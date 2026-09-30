"""Where a 2->2 model's error sits: the per-event validation residual of one run, binned in sqrt(s)
and cos(theta) (the angle between the first beam, row 0, and the first final-state particle, row 2),
as the mean squared error of log|M|^2 per bin. Needs a run made with
evaluation.save_predictions=true (preds_val.npz in the run dir). The predictions come out in the
validation pool's row order; the match is checked on the raw |M|^2 before anything is binned.
Runs on the site that holds the run (numpy only) and prints one JSON line.
    python analysis/transfer/residual_map.py <run_dir> [--bins 8]
"""
import argparse, json, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--bins", type=int, default=8)
a = ap.parse_args()

log = open(os.path.join(a.run_dir, "out_0.log")).read()
m = re.search(r"\[val\] (\S+): (\d+) events.*? from (\S+\.npy)", log)
name, n_val, val_path = m.group(1), int(m.group(2)), m.group(3)
P = np.load(os.path.join(a.run_dir, "preds_val.npz"))
std = json.load(open(os.path.join(a.run_dir, "data_stats.json")))["prepd_std"][0]
pool = np.asarray(np.load(val_path, mmap_mode="r")[:n_val])
npart = (pool.shape[1] - 1) // 5
mom = pool[:, :npart * 4].reshape(-1, npart, 4)
raw_file = pool[:, -1]
raw_in_order = np.asarray(P["raw_truth"]).reshape(-1)
match = float(np.max(np.abs(raw_in_order - raw_file) / np.abs(raw_file)))
assert match < 1e-5, f"prediction order does not match the val pool (max rel diff {match:.2g})"

res = (np.asarray(P["pred"]).reshape(len(raw_file), -1)[:, 0]
       - np.asarray(P["truth"]).reshape(len(raw_file), -1)[:, 0]) * std    # residual in log|M|^2
sq = res ** 2
p0, p2 = mom[:, 0], mom[:, 2]
tot = mom[:, 0] + mom[:, 1]
sqrts = np.sqrt(np.maximum(tot[:, 0] ** 2 - (tot[:, 1:] ** 2).sum(1), 0))
cos = (p0[:, 1:] * p2[:, 1:]).sum(1) / (np.linalg.norm(p0[:, 1:], axis=1) * np.linalg.norm(p2[:, 1:], axis=1))
logm = np.log(np.abs(raw_file))

se = np.geomspace(sqrts.min(), sqrts.max() * (1 + 1e-9), a.bins + 1)
ce = np.linspace(-1, 1 + 1e-9, a.bins + 1)
mse = np.full((a.bins, a.bins), np.nan)
cnt = np.zeros((a.bins, a.bins), int)
share = np.zeros((a.bins, a.bins))
si, ci = np.digitize(sqrts, se) - 1, np.digitize(cos, ce) - 1
for i in range(a.bins):
    for j in range(a.bins):
        sel = (si == i) & (ci == j)
        cnt[i, j] = sel.sum()
        if sel.any():
            mse[i, j] = sq[sel].mean()
            share[i, j] = sq[sel].sum() / sq.sum()
top = np.argsort(sq)[::-1][:20]
print(json.dumps(dict(
    process=name, n=int(len(sq)), mse=float(sq.mean()), order_match=match,
    sqrts_edges=se.tolist(), cos_edges=ce.tolist(), mse_map=np.nan_to_num(mse, nan=-1).tolist(),
    count_map=cnt.tolist(), share_map=share.tolist(),
    share_top1pct=float(np.sort(sq)[::-1][:max(1, len(sq) // 100)].sum() / sq.sum()),
    worst=[dict(sqrts=float(sqrts[i]), cos=float(cos[i]), logm=float(logm[i]), res=float(res[i])) for i in top])))
