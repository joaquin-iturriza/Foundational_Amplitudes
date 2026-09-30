"""ee->WW's forward corner: is the error where the t-channel factor's |t| floor clips, and how many
training events reach that corner at each D? For the nu_e exchange (e- to W-), t = (p_e- - p_W-)^2;
the factor divides |t| out, floored at the train pool's 1e-3 quantile of |t| (experiment.py). Per
validation event the squared residual of log|M|^2 comes from preds_val.npz, matched to pool rows by
the raw |M|^2 as in residual_map.py. Prints one JSON line:
  floor, median |t|; per D the train events below the floor and within 3x and 10x of it; the share of
  the validation squared error carried by the events in those |t| bands; the 20 worst validation
  events with |t|/floor and the percentile of their log|M|^2 in the train pool.
Runs on the site that holds the run (numpy only).
    python analysis/transfer/ww_corner.py <run_dir>
"""
import argparse, json, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
a = ap.parse_args()
log = open(os.path.join(a.run_dir, "out_0.log")).read()
tr_path = re.search(r"\[train\] ee_WW: \d+ events from (\S+\.npy)", log).group(1)
m = re.search(r"\[val\] ee_WW: (\d+) events.*? from (\S+\.npy)", log)
n_val, va_path = int(m.group(1)), m.group(2)


def load(path, n=None):
    x = np.asarray(np.load(path, mmap_mode="r")[:n] if n else np.load(path))
    npart = (x.shape[1] - 1) // 5
    mom = x[:, :npart * 4].reshape(-1, npart, 4).astype(np.float64)
    pid = x[0, npart * 4:npart * 5].astype(int)
    return mom, pid, x[:, -1].astype(np.float64)


def mandelstam_t(mom, pid):
    q = mom[:, list(pid).index(11)] - mom[:, list(pid).index(-24)]
    return q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1)


def sqrts_cos(mom):
    tot = mom[:, 0] + mom[:, 1]
    s = np.sqrt(np.maximum(tot[:, 0] ** 2 - (tot[:, 1:] ** 2).sum(1), 0))
    p0, p2 = mom[:, 0, 1:], mom[:, 2, 1:]
    return s, (p0 * p2).sum(1) / (np.linalg.norm(p0, axis=1) * np.linalg.norm(p2, axis=1))


mt, pid, amp_tr = load(tr_path)
at_tr = np.abs(mandelstam_t(mt, pid))
floor, med = float(np.quantile(at_tr, 1e-3)), float(np.median(at_tr))
per_D = {}
for k in range(2, 11):
    D = int(round(10 ** (k / 2)))
    x = at_tr[:D]
    per_D[D] = dict(below=int((x < floor).sum()), within3=int((x < 3 * floor).sum()), within10=int((x < 10 * floor).sum()))

mv, _, raw_val = load(va_path, n_val)
at_va = np.abs(mandelstam_t(mv, pid))
P = np.load(os.path.join(a.run_dir, "preds_val.npz"))
std = json.load(open(os.path.join(a.run_dir, "data_stats.json")))["prepd_std"][0]
raw_pred_rows = np.asarray(P["raw_truth"]).reshape(-1)
i_file, i_pred = np.argsort(raw_val), np.argsort(raw_pred_rows)
match = float(np.max(np.abs(raw_pred_rows[i_pred] - raw_val[i_file]) / np.abs(raw_val[i_file])))
assert match < 1e-3, f"predictions do not match the val pool (max rel diff {match:.2g})"
row_of = np.empty(len(raw_val), int); row_of[i_pred] = i_file
res = (np.asarray(P["pred"]).reshape(len(raw_val), -1)[:, 0] - np.asarray(P["truth"]).reshape(len(raw_val), -1)[:, 0]) * std
sq = res ** 2
at_p, raw_p = at_va[row_of], raw_val[row_of]
sq_s, cos_p = sqrts_cos(mv[row_of])
share = {f"below_{f}x": dict(n=int((at_p < f * floor).sum()), share=float(sq[at_p < f * floor].sum() / sq.sum()))
         for f in (1, 3, 10)}
lt = np.sort(np.log(amp_tr))
worst = [dict(sqrts=float(sq_s[i]), cos=float(cos_p[i]), t_over_floor=float(at_p[i] / floor), res=float(res[i]),
              logm_pct_train=float(100 * np.searchsorted(lt, np.log(raw_p[i])) / len(lt)))
         for i in np.argsort(sq)[::-1][:20]]
print(json.dumps(dict(run_dir=os.path.abspath(a.run_dir), floor=floor, median=med, t_range=[float(at_tr.min()), float(at_tr.max())],
                      per_D=per_D, val_bands=share, mse=float(sq.mean()), order_match=match, worst=worst)))
