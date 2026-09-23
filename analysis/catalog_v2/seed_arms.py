"""Arms with repeats: each arm is a glob of run dirs (one per seed). Per arm, the combined
validation loss (mean and spread over seeds), and per class the median and 90th percentile
of the per-process final loss with its spread over seeds; figure: per-class ECDF per arm
with the seed band.
    python analysis/catalog_v2/seed_arms.py "baseline=runs/t1000_slq1e-2_s*" "slq 1e-3=runs/t1000_slq1e-3_s*" ... [--out=name]
The first arm is the baseline. Writes analysis/catalog_v2/<out>_a ... _f (png+pdf, default
seed_arms), one panel per class in the order of CLASSES, the class as the legend title."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import census as C
import plot_style as ps
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
nseeds = max(len(r) for r in data.values())
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
# figure: one panel per class
figs = ps.panels(len(CLASSES))
cols = [ps.C.blue, ps.C.vermillion, ps.C.green, ps.C.orange, ps.C.purple]
grid = np.logspace(-4, 1, 200)
for (fig, ax), c in zip(figs, CLASSES):
    lo_all, hi_all = [], []
    for (l, runs), col in zip(data.items(), cols):
        ecdfs = []
        for _, p in runs:
            v = np.sort([p[n] for n in names if cls(n) == c]); ecdfs.append(np.searchsorted(v, grid, side="right") / len(v))
            lo_all.append(v[0]); hi_all.append(v[-1])
        ecdfs = np.array(ecdfs); ax.plot(grid, ecdfs.mean(0), color=col, label=l)
        if len(runs) > 1: ax.fill_between(grid, ecdfs.min(0), ecdfs.max(0), color=col, alpha=0.2)
    if nseeds > 1:   # legend entry for the shaded bands (an empty artist carrying the label)
        ax.fill_between([], [], [], color=ps.C.grey, alpha=0.2, label=f"min to max over {nseeds} seeds")
    ax.set_xscale("log"); ax.set_xlim(min(lo_all) / 2, max(hi_all) * 2)
    ax.set_xlabel(r"final validation MSE($\log|\mathcal{M}|^2$) per process")
    ax.set_ylabel("fraction of processes")
    # the arm names are too long for a legend inside a 2.4in box: one strip above the plot
    ps.process_label(ax, f"{C.CLASS_LABEL[c]} ({sum(cls(n) == c for n in names)})", loc="upper left")
    ps.shared_legend(fig, ax, ncol=1)
ps.save_panels(figs, f"analysis/catalog_v2/{opts.get('out', 'seed_arms')}")
