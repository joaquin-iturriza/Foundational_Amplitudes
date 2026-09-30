"""Where a 2->2 model's error sits: the per-event validation residual of one run, binned in sqrt(s)
and cos(theta) (the angle between the first beam, row 0, and the first final-state particle, row 2),
as the mean squared error of log|M|^2 per bin. Needs a run made with
evaluation.save_predictions=true (preds_val.npz in the run dir). The predictions come out shuffled;
each is matched to its validation-pool row by rank of the raw |M|^2, and the match is checked before
anything is binned. Rank matching can only swap two events whose |M|^2 differ by less than the
match precision (the saved raw_truth is float32): those are counted (n_ambiguous, the events with a
neighbour in |M|^2 closer than twice the largest matching error) and the map is refused above 1%.
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
# the events are shuffled when the splits are assembled (experiment.py, seed 42): match each
# prediction to its pool row by the raw |M|^2, which is distinct per event, and check the match
raw_pred_rows = np.asarray(P["raw_truth"]).reshape(-1)
i_file, i_pred = np.argsort(raw_file), np.argsort(raw_pred_rows)
match = float(np.max(np.abs(raw_pred_rows[i_pred] - raw_file[i_file]) / np.abs(raw_file[i_file])))
assert match < 1e-3, f"predictions do not match the val pool (max rel diff {match:.2g})"
row_of = np.empty(len(raw_file), int); row_of[i_pred] = i_file          # prediction k -> pool row
gaps = np.diff(np.log(np.abs(raw_file[i_file])))
close = gaps < 2 * match                                                    # a swap is possible here
amb = np.zeros(len(raw_file), bool); amb[:-1] |= close; amb[1:] |= close
n_amb = int(amb.sum())
assert n_amb <= 0.01 * len(raw_file), f"{n_amb} events are closer in |M|^2 than the match precision"
mom, raw_file = mom[row_of], raw_file[row_of]                              # pool rows in prediction order

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
    run_dir=os.path.abspath(a.run_dir), process=name, n=int(len(sq)), mse=float(sq.mean()),
    order_match=match, n_ambiguous=n_amb,
    sqrts_edges=se.tolist(), cos_edges=ce.tolist(), mse_map=np.nan_to_num(mse, nan=-1).tolist(),
    count_map=cnt.tolist(), share_map=share.tolist(),
    share_top1pct=float(np.sort(sq)[::-1][:max(1, len(sq) // 100)].sum() / sq.sum()),
    worst=[dict(sqrts=float(sqrts[i]), cos=float(cos[i]), logm=float(logm[i]), res=float(res[i])) for i in top])))
