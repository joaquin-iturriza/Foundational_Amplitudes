"""Re-aggregate IG attribution results over a chosen subset of processes.

The multi-process baseline is well-converged on only a few processes; attributions
on badly-fit processes are meaningless and contaminate the global ranking. This
re-pools the per-process attributions (saved in ig_results.npz) over a fit-quality
cut, on CPU. No model evaluation needed."""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ap = argparse.ArgumentParser()
_ap.add_argument("--run-dir", default="runs/allopts_baseline")
_ap.add_argument("--mse-cut", type=float, default=5e-3)
_args = _ap.parse_args()
NPZ = os.path.join(_args.run_dir, "attribution", "ig_results.npz")
OUT = os.path.join(_args.run_dir, "attribution", "ig_goodfit")
MSE_CUT = _args.mse_cut   # keep processes the model actually learned

d = np.load(NPZ, allow_pickle=True)
prop_names = list(d["prop_names"])
procs = [k[:-5] for k in d.files if k.endswith("__mse")]

rows = []
for p in procs:
    mse = float(d[f"{p}__mse"])
    prop = d[f"{p}__prop"]                 # (8,) mean|attr| per property dim
    mom_comp = d[f"{p}__mom_comp"]         # (4,) mean|attr| per momentum component
    P = len(d[f"{p}__mom_slot"])
    rows.append(dict(name=p, mse=mse, prop=prop, mom_comp=mom_comp, P=P))
rows.sort(key=lambda r: r["mse"])

print(f"{'process':<34s} {'MSE':>10s}  fit?")
for r in rows:
    print(f"{r['name']:<34s} {r['mse']:>10.2e}  {'KEEP' if r['mse']<MSE_CUT else 'drop'}")

def aggregate(subset):
    # weight each process by its particle count P (all have equal event count n=256)
    W = np.array([r["P"] for r in subset], dtype=float)
    prop = np.stack([r["prop"] for r in subset])          # (k,8)
    momc = np.stack([r["mom_comp"] for r in subset])      # (k,4)
    prop_g = np.average(prop, axis=0, weights=W)          # (8,)
    momc_g = np.average(momc, axis=0, weights=W)          # (4,)
    # per-scalar group importance: average |attr| over the scalars in each group
    mom_scalar = momc_g.mean()
    prop_scalar = prop_g.mean()
    return prop_g, momc_g, np.array([mom_scalar, prop_scalar, 0.0])

good = [r for r in rows if r["mse"] < MSE_CUT]
allp = rows
prop_good, momc_good, grp_good = aggregate(good)
prop_all, momc_all, grp_all = aggregate(allp)

print(f"\n=== property importance, GOOD-FIT only ({', '.join(r['name'].split('_')[1] for r in good)}) ===")
for i in np.argsort(prop_good)[::-1]:
    print(f"  {prop_names[i]:>16s} : {prop_good[i]:.4e}   (all-proc: {prop_all[i]:.4e})")

print("\n=== momentum components (good-fit), mean|IG| ===")
for nm, v in zip(["E", "px", "py", "pz"], momc_good):
    print(f"  {nm:>3s} : {v:.4e}")

print("\n=== input-group per-scalar mean|IG| (good-fit) ===")
print(f"  momentum={grp_good[0]:.4e}  properties={grp_good[1]:.4e}  order=0")

# ---- plots ----
fig, ax = plt.subplots(figsize=(7, 4))
o = np.argsort(prop_good)[::-1]
x = np.arange(len(prop_names))
ax.bar(x, prop_good[o], color="#3b6ea5")
ax.set_xticks(x); ax.set_xticklabels([prop_names[i] for i in o], rotation=40, ha="right")
ax.set_ylabel("mean |IG attribution|")
ax.set_title(f"Physical-property importance (well-fit processes, MSE<{MSE_CUT:g})")
fig.tight_layout(); fig.savefig(f"{OUT}_property_importance.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["momentum\n(kinematics)", "properties\n(identity)", "coupling\norder"],
       grp_good, color=["#c0504d", "#3b6ea5", "#9bbb59"])
ax.set_ylabel("per-scalar mean |IG attribution|")
ax.set_title("Input-group importance (well-fit processes)")
fig.tight_layout(); fig.savefig(f"{OUT}_group_importance.pdf"); plt.close(fig)

print(f"\nSaved {OUT}_property_importance.pdf and {OUT}_group_importance.pdf")
