"""Arms with repeats: each arm is a glob of run dirs (one per seed). Per arm, the combined
validation loss (mean and spread over seeds), and per class the median and 90th percentile
of the per-process final loss with its spread over seeds; figure: per-class ECDF per arm
with the seed band, and the per-process loss of each arm against the baseline (seed-mean).
    python analysis/catalog_v2/seed_arms.py "baseline=runs/t1000_slq1e-2_s*" "slq 1e-3=runs/t1000_slq1e-3_s*" ... [--out=name] [--title=...]
The first arm is the baseline. Writes analysis/catalog_v2/<out>.{png,pdf} (default seed_arms)."""
import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import census as C
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
arms = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in sys.argv[1:] if not a.startswith("--")]
NP = json.load(open(os.path.join(HERE, "n_particles.json"))); s27, all50 = C.signed_classes()
def cls(n):
    if n in all50: return "signed 1-loop"
    if n.endswith("_nlo") or n.endswith("_loop"): return "positive 1-loop"
    if n in C.NEEDLE or "__mz" in n: return "resonant 2->2"
    return f"tree 2->{NP[n]-2}"
CLASSES = ["tree 2->2", "resonant 2->2", "tree 2->3", "tree 2->4", "positive 1-loop", "signed 1-loop"]
def load(pattern):
    out = []
    for r in sorted(glob.glob(pattern)):
        js = sorted(glob.glob(os.path.join(r, "**", "per_process_metrics.json"), recursive=True))
        if not js: continue
        d = json.load(open(js[-1]))
        out.append((d["val_loss_no_reg"][-1], {n: v[-1] for n, v in d["proc_val_losses_no_reg"].items() if v and n in NP}))
    return out
data = {label: load(pat) for label, pat in arms}
names = sorted(set.intersection(*[set(p.keys()) for runs in data.values() for _, p in runs]))
print(f"{len(names)} processes; arms: " + ", ".join(f"{l} ({len(r)} seeds)" for l, r in data.items()))
print(f"{'arm':22s} combined (mean ± spread over seeds)")
for l, runs in data.items():
    c = np.array([x for x, _ in runs]); print(f"{l:22s} {c.mean():.4f} ± {c.std(ddof=1) if len(c) > 1 else 0:.4f}   [{', '.join(f'{v:.4f}' for v in c)}]")
print(f"\n{'class':16s} " + " | ".join(f"{l:>26s}" for l in data))
print(f"{'':16s} " + " | ".join(f"{'median (±seed) / 90%':>26s}" for _ in data))
for c in CLASSES:
    cells = []
    for l, runs in data.items():
        meds = [np.median([p[n] for n in names if cls(n) == c]) for _, p in runs]
        p90 = [np.percentile([p[n] for n in names if cls(n) == c], 90) for _, p in runs]
        cells.append(f"{np.mean(meds):.3g} (±{np.std(meds, ddof=1) if len(meds) > 1 else 0:.2g}) / {np.mean(p90):.3g}")
    print(f"{c:16s} " + " | ".join(f"{x:>26s}" for x in cells))
# figure
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5)); axes = axes.ravel()
for k, c in enumerate(CLASSES):
    ax = axes[k]
    for l, runs in data.items():
        grid = np.logspace(-4, 1, 200); ecdfs = []
        for _, p in runs:
            v = np.sort([p[n] for n in names if cls(n) == c]); ecdfs.append(np.searchsorted(v, grid, side="right") / len(v))
        ecdfs = np.array(ecdfs); line, = ax.plot(grid, ecdfs.mean(0), label=l)
        if len(runs) > 1: ax.fill_between(grid, ecdfs.min(0), ecdfs.max(0), color=line.get_color(), alpha=0.2)
    ax.set_xscale("log"); ax.set_title(f"{c} ({sum(cls(n) == c for n in names)})", fontsize=10); ax.grid(alpha=0.3)
    ax.set_xlim(1e-4, 1e1); ax.set_xlabel("final validation loss (per process)")
axes[0].legend(fontsize=8); axes[0].set_ylabel("fraction of processes"); axes[3].set_ylabel("fraction of processes")
fig.suptitle(opts.get("title", "arms with repeats: per-class ECDF, band = min/max over seeds"), fontsize=12); fig.tight_layout()
base = os.path.join(HERE, opts.get("out", "seed_arms")); fig.savefig(base + ".png", dpi=130); fig.savefig(base + ".pdf"); print("wrote", base + ".{png,pdf}")
