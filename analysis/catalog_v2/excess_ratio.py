"""Distance of every process from its multiplicity reference, e_p = m_p / L_ref(n_p), without a
pass/fail threshold: figure e_p against the pool's ln|M|^2 range per arm, quantiles by class,
and the processes furthest above their reference with their attributes.
    python analysis/catalog_v2/excess_ratio.py --ref=4=1.66e-3_5=1.03e-2_6=1.81e-2 label=run_dir [label=run_dir ...]
Writes analysis/catalog_v2/excess_ratio.{png,pdf}."""
import csv, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import census as C
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
REF = {int(k): float(v) for k, v in (t.split("=") for t in opts["ref"].split("_"))}
runs = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in sys.argv[1:] if not a.startswith("--")]
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
aud = {re.sub(r"_\d+-\d+GeV_train(_smix)?$", "", r["name"]): r for r in csv.DictReader(open(os.path.join(HERE, "pool_audit.csv"))) if r["role"] == "train"}
s27, all50 = C.signed_classes()
def cls(n):
    if n in all50: return "signed 1-loop"
    if n.endswith("_nlo") or n.endswith("_loop"): return "positive 1-loop"
    if n in C.NEEDLE or "__mz" in n: return "resonant 2->2"
    return f"tree 2->{NP[n]-2}"
fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4), sharey=True)
axes = np.atleast_1d(axes)
for ax, (label, path) in zip(axes, runs):
    d = C.read_run(path)[1][-1][1]
    names = [n for n in d if n in aud and n in NP]
    e = {n: d[n] / REF[NP[n]] for n in names}
    sp = np.array([float(aud[n]["logspread"]) for n in names]); ev = np.array([e[n] for n in names]); npart = np.array([NP[n] for n in names])
    for k, mk in ((4, "o"), (5, "s"), (6, "^")):
        sel = npart == k; ax.scatter(sp[sel], ev[sel], s=14, marker=mk, alpha=0.7, label=f"2$\\to${k-2}")
    ax.axhline(1, color="k", lw=0.8, ls="--"); ax.set_yscale("log"); ax.set_xlabel("range of ln|M|^2 in the train pool"); ax.grid(alpha=0.3)
    ax.set_title(f"{label}: median e = {np.median(ev):.2g}", fontsize=10)
    print(f"\n== {label}: e_p = m_p / L_ref(n_p); median {np.median(ev):.2g}, quartiles {np.percentile(ev,25):.2g}-{np.percentile(ev,75):.2g}, below 1: {np.mean(ev<1):.0%}, above 10: {np.mean(ev>10):.0%}")
    groups = {}
    for n in names: groups.setdefault(cls(n), []).append(e[n])
    for g, v in sorted(groups.items()): v = np.array(v); print(f"   {g:16s} n={len(v):3d}  median {np.median(v):6.2g}  90% {np.percentile(v,90):6.2g}  max {v.max():6.2g}")
    worst = sorted(names, key=lambda n: -e[n])[:15]
    print("   furthest above reference: " + ", ".join(f"{n}({e[n]:.0f}, {cls(n)}, range {float(aud[n]['logspread']):.0f})" for n in worst))
axes[0].set_ylabel("m_p / L_ref(multiplicity)"); axes[0].legend(title="multiplicity", fontsize=8)
fig.tight_layout(); base = os.path.join(HERE, "excess_ratio"); fig.savefig(base + ".png", dpi=150); fig.savefig(base + ".pdf")
