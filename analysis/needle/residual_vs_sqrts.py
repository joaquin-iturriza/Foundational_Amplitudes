"""Where does the s-channel 2->2 error live? Bin the final-step validation residual of a run
(preds/<step>_<dataset>_val_preprocessed_pred.npy, standardized log|M|^2) in sqrt(s) against
the frozen val pool (flat sampling, the physical measure). Writes a table and a figure
(png+pdf) per run: MSE per sqrt(s) bin, the pole bin (|sqrt(s)-M_Z|<10 GeV) share of the
total MSE, and the mean signed residual near the pole (peak height bias)."""
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import datagen
MZ = 91.1876
run_dir = sys.argv[1]; out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(run_dir.rstrip("/")))
os.makedirs(out_dir, exist_ok=True)
stats = json.load(open(os.path.join(run_dir, "data_stats.json")))
import yaml
cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
names = [p["name"] for p in yaml.safe_load(open(cfg["data"]["processes_file"]))["processes"]]
means, stds = stats["prepd_mean"], stats["prepd_std"]
preds = sorted(glob.glob(os.path.join(run_dir, "preds", "*_val_preprocessed_pred.npy")))
edges = np.array([25, 40, 60, 75, 85, 88, 90, 91, 92, 94, 97, 105, 130, 200, 400, 700, 1000], float)
rows = []
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
for i, name in enumerate(names):
    pf = [p for p in preds if re.search(rf"_{re.escape(name)}_val_", os.path.basename(p))]
    if not pf: print("no preds for", name); continue
    pred = np.load(pf[-1]).ravel()
    vf = glob.glob(os.path.join(datagen.frozen_dir(), f"{name}_*_val_amplitudes.npy"))[0]
    a = np.asarray(np.load(vf, mmap_mode="r")[:len(pred)]); n = (a.shape[1] - 1) // 5
    P = a[:, :4 * n].reshape(-1, n, 4); q = P[:, 0] + P[:, 1]; sq = np.sqrt(np.maximum(q[:, 0]**2 - (q[:, 1:]**2).sum(1), 0))
    z = (np.log(a[:, -1]) - means[i]) / stds[i]                      # the standardized target as the trainer sees it
    r = pred - z; mse = np.mean(r**2)
    pole = np.abs(sq - MZ) < 10
    share = np.sum(r[pole]**2) / np.sum(r**2)
    b = np.digitize(sq, edges) - 1
    binned = [np.mean(r[b == k]**2) if np.any(b == k) else np.nan for k in range(len(edges) - 1)]
    rows.append((name, mse, pole.mean(), share, np.mean(r[pole]), np.mean(np.abs(r[~pole])), z[pole].mean(), z[pole].max()))
    ctr = 0.5 * (edges[1:] + edges[:-1])
    axes[0].plot(ctr, binned, marker="o", ms=3, label=name); axes[1].plot(ctr, [np.mean(r[b == k]) if np.any(b == k) else np.nan for k in range(len(edges) - 1)], marker="o", ms=3)
axes[0].set_yscale("log"); axes[0].set_xscale("log"); axes[0].set_ylabel("val MSE per bin (standardized log)"); axes[0].legend(fontsize=7, ncol=2); axes[0].axvline(MZ, color="k", lw=0.5)
axes[1].set_ylabel("mean signed residual (pred - target)"); axes[1].set_xlabel("sqrt(s) [GeV]"); axes[1].axhline(0, color="k", lw=0.5); axes[1].axvline(MZ, color="k", lw=0.5)
fig.tight_layout(); base = os.path.join(out_dir, "residual_vs_sqrts"); fig.savefig(base + ".png", dpi=130); fig.savefig(base + ".pdf")
print(f"{'process':10s} {'MSE':>8s} {'pole frac':>9s} {'pole share of MSE':>17s} {'mean resid @pole':>16s} {'mean|resid| off':>15s} {'z@pole mean/max':>16s}")
for r in rows: print(f"{r[0]:10s} {r[1]:8.4f} {r[2]:9.3f} {r[3]:17.2f} {r[4]:+16.3f} {r[5]:15.3f} {r[6]:+7.2f}/{r[7]:+5.2f}")
print("figure:", base + ".png/.pdf")
