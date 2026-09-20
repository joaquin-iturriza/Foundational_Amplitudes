"""Fit the per-multiplicity reference curve of the excess aggregation from solo runs.

    python analysis/catalog_v2/fit_excess_reference.py runs/ref_solo_ee_uu runs/ref_solo_ee_uug runs/ref_solo_ee_uugg

Each run trains one process alone at the joint run's per-process batch (bs/P). Its
validation history L(t) (val_loss_no_reg of the process at step t) is fitted with
L(t) = A t^-alpha + Linf (grid over Linf, linear fit in log space), keyed by the process's
particle count. Prints the training.excess_reference string for the job scripts and writes
analysis/catalog_v2/excess_reference.{json,png,pdf}."""
import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NP = json.load(open(os.path.join(HERE, "n_particles.json")))


def load(run):
    f = sorted(glob.glob(os.path.join(run, "**", "per_process_metrics.json"), recursive=True))[-1]
    d = json.load(open(f)); name = d["dataset_order"][0]
    L = np.array(d["proc_val_losses_no_reg"][name], dtype=float)
    t = (np.arange(len(L)) + 1) * d["validate_every_n_steps"]
    return name, t, L


def fit(t, L):
    best = None
    for linf in np.concatenate([[0.0], np.geomspace(L.min() * 1e-3, L.min() * 0.99, 60)]):
        y = np.log(np.maximum(L - linf, 1e-12)); X = np.stack([np.ones_like(t, dtype=float), -np.log(t)], 1)
        coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = float(((X @ coef - y) ** 2).sum())
        if best is None or r < best[0]:
            best = (r, float(np.exp(coef[0])), float(coef[1]), float(linf))
    return best[1:]


out = {}; fig, ax = plt.subplots(figsize=(6, 4))
for run in sys.argv[1:]:
    name, t, L = load(run); n = NP[name]
    A, alpha, linf = fit(t, L); out[n] = dict(A=A, alpha=alpha, Linf=linf, process=name)
    ax.plot(t, L, "o", ms=3, label=f"{name} ({n} particles)")
    tt = np.geomspace(t[0], t[-1] * 3, 100); ax.plot(tt, A * tt ** (-alpha) + linf, "-", lw=1)
    print(f"{name:10s} n={n}  A={A:.3g} alpha={alpha:.3f} Linf={linf:.3g}  L(1000)={A*1000**-alpha+linf:.3g}")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("step (solo, bs = joint bs / P)"); ax.set_ylabel("validation loss"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
base = os.path.join(HERE, "excess_reference"); fig.tight_layout(); fig.savefig(base + ".png", dpi=150); fig.savefig(base + ".pdf")
json.dump(out, open(base + ".json", "w"), indent=1)
ref = "{" + ",".join(f"{n}:{{A:{c['A']:.4g},alpha:{c['alpha']:.4f},Linf:{c['Linf']:.4g}}}" for n, c in sorted(out.items())) + "}"
print("REF=" + ref)
