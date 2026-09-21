"""The aggregation arms side by side, one column per arm: the combined validation curve, every
process's validation curve, the final loss against the pool's ln|M|^2 range, and the final
loss by class (ECDF). Reads plots_0/per_process_metrics.json of each run.
    python analysis/catalog_v2/arms_compare.py "label=runs/<run>" ["label=runs/<run>" ...] [--out=name] [--title=...]
Writes analysis/catalog_v2/<out>.{png,pdf} (default arms_compare)."""
import csv, glob, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import census as C
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
runs = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in sys.argv[1:] if not a.startswith("--")]
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
aud = {re.sub(r"_\d+-\d+GeV_train(_smix)?$", "", r["name"]): r for r in csv.DictReader(open(os.path.join(HERE, "pool_audit.csv"))) if r["role"] == "train"}
s27, all50 = C.signed_classes()
def cls(n):
    if n in all50: return "signed 1-loop"
    if n.endswith("_nlo") or n.endswith("_loop"): return "positive 1-loop"
    if n in C.NEEDLE or "__mz" in n: return "resonant 2->2"
    return f"tree 2->{NP[n]-2}"
CLASSES = ["tree 2->2", "resonant 2->2", "tree 2->3", "tree 2->4", "positive 1-loop", "signed 1-loop"]
COL = {4: "C0", 5: "C1", 6: "C2"}
fig, axes = plt.subplots(4, len(runs), figsize=(4.6 * len(runs), 14), squeeze=False)
for j, (label, path) in enumerate(runs):
    js = sorted(glob.glob(os.path.join(path, "**", "per_process_metrics.json"), recursive=True))[-1]
    d = json.load(open(js)); every = d["validate_every_n_steps"]
    curves = d["proc_val_losses_no_reg"]; final = {n: v[-1] for n, v in curves.items() if v}
    steps = every * np.arange(1, len(d["val_loss_no_reg"]) + 1)
    ax = axes[0, j]; ax.plot(steps, d["val_loss_no_reg"], "k-"); ax.set_yscale("log"); ax.set_xlabel("step")
    ax.set_title(f"{label}\ncombined validation (geometric mean): final {d['val_loss_no_reg'][-1]:.3g}", fontsize=10); ax.grid(alpha=0.3)
    ax = axes[1, j]
    for n, v in curves.items():
        if n in NP: ax.plot(every * np.arange(1, len(v) + 1), v, color=COL[NP[n]], lw=0.4, alpha=0.35)
    for k in (4, 5, 6): ax.plot([], [], color=COL[k], label=f"2->{k-2}")
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_title("every process's validation loss", fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[2, j]
    names = [n for n in final if n in aud and n in NP]
    sp = np.array([float(aud[n]["logspread"]) for n in names]); y = np.array([final[n] for n in names]); npart = np.array([NP[n] for n in names])
    for k, mk in ((4, "o"), (5, "s"), (6, "^")):
        sel = npart == k; ax.scatter(sp[sel], y[sel], s=12, marker=mk, alpha=0.7, color=COL[k], label=f"2->{k-2}")
    ax.axhline(0.05, color="k", lw=0.8, ls="--"); ax.set_yscale("log"); ax.set_xlabel("range of ln|M|^2 in the train pool"); ax.grid(alpha=0.3)
    ax.set_title(f"final loss against the target's range: {int((y > 0.05).sum())}/{len(y)} above 0.05", fontsize=10)
    ax = axes[3, j]
    for c in CLASSES:
        v = np.sort([final[n] for n in names if cls(n) == c])
        if len(v): ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", label=f"{c} ({len(v)}, median {np.median(v):.2g})")
    ax.set_xscale("log"); ax.set_xlabel("final validation loss"); ax.set_title("by class (ECDF)", fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=7)
for i in range(3):
    lo = min(a.get_ylim()[0] for a in axes[i]); hi = max(a.get_ylim()[1] for a in axes[i])
    for a in axes[i]: a.set_ylim(lo, hi)
lo = min(a.get_xlim()[0] for a in axes[3]); hi = max(a.get_xlim()[1] for a in axes[3])
for a in axes[3]: a.set_xlim(lo, hi)
axes[0, 0].set_ylabel("validation loss"); axes[1, 0].set_ylabel("validation loss (per process)"); axes[2, 0].set_ylabel("final validation loss"); axes[3, 0].set_ylabel("fraction of processes")
fig.suptitle(opts.get("title", "catalog_v2 aggregation arms"), fontsize=12); fig.tight_layout()
base = os.path.join(HERE, opts.get("out", "arms_compare")); fig.savefig(base + ".png", dpi=130); fig.savefig(base + ".pdf"); print("wrote", base + ".{png,pdf}")
