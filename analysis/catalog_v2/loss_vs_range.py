"""Per-process validation loss against the pool's log|M|^2 range, two aggregations.
    python analysis/catalog_v2/loss_vs_range.py runs/<geo run> runs/<mean run>
Writes analysis/catalog_v2/loss_vs_range.{png,pdf}."""
import csv, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import census as C
G = C.read_run(sys.argv[1])[1][-1][1]; M = C.read_run(sys.argv[2])[1][-1][1]
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
aud = {re.sub(r"_\d+-\d+GeV_train(_smix)?$", "", r["name"]): r
       for r in csv.DictReader(open(os.path.join(HERE, "pool_audit.csv"))) if r["role"] == "train"}
names = [n for n in G if n in aud and n in NP]
sp = np.array([float(aud[n]["logspread"]) for n in names]); g = np.array([G[n] for n in names]); m = np.array([M[n] for n in names])
npart = np.array([NP[n] for n in names])
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, y, title in ((axes[0], g, "geometric mean"), (axes[1], m, "arithmetic mean")):
    for k, mk in ((4, "o"), (5, "s"), (6, "^")):
        sel = npart == k
        ax.scatter(sp[sel], y[sel], s=14, marker=mk, alpha=0.7, label=f"2$\\to${k-2}")
    ax.axhline(0.05, color="k", lw=0.8, ls="--"); ax.set_yscale("log"); ax.set_xlabel("range of ln|M|^2 in the train pool")
    ax.set_title(f"{title}: {int((y > 0.05).sum())}/{len(y)} above 0.05"); ax.grid(alpha=0.3)
axes[0].set_ylabel("validation MSE (standardized target)"); axes[0].legend(title="multiplicity", fontsize=8)
fig.suptitle("catalog_v2, 1000 steps, shaped measure: loss against the target's dynamic range")
fig.tight_layout()
base = os.path.join(HERE, "loss_vs_range"); fig.savefig(base + ".png", dpi=150); fig.savefig(base + ".pdf")
print("wrote", base + ".{png,pdf}")
