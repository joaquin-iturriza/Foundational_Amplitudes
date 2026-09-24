"""Per-process validation loss at the best checkpoint against the pool's ln|M|^2 range, one panel per run.
    python analysis/catalog_v2/loss_vs_range.py runs/<run A> runs/<run B> [--labels=A,B] [--out=name]
Writes analysis/catalog_v2/<out>_a, _b, ... (png+pdf), one panel per run, the run's label as the
legend title. No pass/fail line: the loss is read against the range, not against a threshold."""
import csv, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import census as C
import plot_style as ps
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
labels = opts.get("labels", "geometric mean,arithmetic mean").split(",")
outname = opts.get("out", "loss_vs_range")
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
aud = {re.sub(r"_\d+-\d+GeV_train(_smix)?$", "", r["name"]): r
       for r in csv.DictReader(open(os.path.join(HERE, "pool_audit.csv"))) if r["role"] == "train"}
MULT = ((4, "o", ps.C.blue), (5, "s", ps.C.vermillion), (6, "^", ps.C.green))
figs = ps.panels(len(args))
for (fig, ax), path, label in zip(figs, args, labels):
    d = C.at_best(C.metrics(path))[2]   # the best checkpoint
    names = [n for n in d if n in aud and n in NP]
    sp = np.array([float(aud[n]["logspread"]) for n in names]); y = np.array([d[n] for n in names])
    npart = np.array([NP[n] for n in names])
    for k, mk, col in MULT:
        sel = npart == k
        ax.scatter(sp[sel], y[sel], marker=mk, color=col, alpha=0.7, label=rf"$2\to{k-2}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"range of $\ln|\mathcal{M}|^2$ in the train pool")
    ax.set_ylabel(r"validation MSE($\log|\mathcal{M}|^2$)")
    ps.process_label(ax, label, loc="upper left")
    ps.shared_legend(fig, ax, ncol=3)
    print(f"{label}: median {np.median(y):.3g}, 90% {np.percentile(y, 90):.3g}")
ps.save_panels(figs, f"analysis/catalog_v2/{outname}")
