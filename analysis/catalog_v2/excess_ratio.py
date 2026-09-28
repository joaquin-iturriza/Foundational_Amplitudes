"""Distance of every process from its multiplicity reference, e_p = m_p / L_ref(n_p) (m_p at the run's
best checkpoint), without a pass/fail threshold: figure e_p against the pool's ln|M|^2 range per arm, quantiles by class,
and the processes furthest above their reference with their attributes.
    python analysis/catalog_v2/excess_ratio.py --solo-steps=S label=run_dir [label=run_dir ...] [--out=name]
L_ref(n) is what multiplicity n reaches alone at the same per-process events seen: the batch-1024 solo
references (solo_b1k.tree_reference) at S steps, the geometric mean over the two tree reference processes
of that multiplicity, each at its sweep's best validation (CLAUDE.md, Reported values). S = 33 is the
grid point of the 1000-step joint runs (33 x 1024 events; a tree process of the full-pool catalog sees
1000 x 16384 x 100k / sum of the pools = 38k).
Writes analysis/catalog_v2/<out>_a, _b, ... (png+pdf, default excess_ratio), one panel per arm,
the arm as the legend title."""
import csv, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import census as C
import plot_style as ps
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
from solo_b1k import tree_reference
REF = tree_reference(int(opts["solo-steps"]))
print("L_ref: " + ", ".join(f"2->{k-2} {v:.3g}" for k, v in sorted(REF.items())))
runs = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in sys.argv[1:] if not a.startswith("--")]
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
aud = {re.sub(r"_\d+-\d+GeV_train(_smix)?$", "", r["name"]): r for r in csv.DictReader(open(os.path.join(HERE, "pool_audit.csv"))) if r["role"] == "train"}
s27, all50 = C.signed_classes()
def cls(n):
    if n in all50: return "signed 1-loop"
    if n.endswith("_nlo") or n.endswith("_loop"): return "positive 1-loop"
    if n in C.NEEDLE or "__mz" in n: return "resonant 2->2"
    return f"tree 2->{NP[n]-2}"
MULT = ((4, "o", ps.C.blue), (5, "s", ps.C.vermillion), (6, "^", ps.C.green))
figs = ps.panels(len(runs))
for (fig, ax), (label, path) in zip(figs, runs):
    d = C.at_best(C.metrics(path))[2]   # the best checkpoint
    names = [n for n in d if n in aud and n in NP]
    e = {n: d[n] / REF[NP[n]] for n in names}
    sp = np.array([float(aud[n]["logspread"]) for n in names]); ev = np.array([e[n] for n in names]); npart = np.array([NP[n] for n in names])
    for k, mk, col in MULT:
        sel = npart == k; ax.scatter(sp[sel], ev[sel], marker=mk, color=col, alpha=0.7, label=rf"$2\to{k-2}$")
    ax.axhline(1, color=ps.C.grey, ls="--", label="alone, same events seen")
    ax.set_yscale("log"); ax.set_xlabel(r"range of $\ln|\mathcal{M}|^2$ in the train pool")
    ax.set_ylabel(r"$\mathrm{MSE}_p\,/\,\mathrm{MSE}_{\rm solo}(n_p)$")
    ps.process_label(ax, label, loc="upper left")
    ps.shared_legend(fig, ax, ncol=2)
    print(f"\n== {label}: e_p = m_p / L_ref(n_p); median {np.median(ev):.2g}, quartiles {np.percentile(ev,25):.2g}-{np.percentile(ev,75):.2g}, below the reference (e_p < 1): {np.mean(ev<1):.0%}")
    groups = {}
    for n in names: groups.setdefault(cls(n), []).append(e[n])
    for g, v in sorted(groups.items()): v = np.array(v); print(f"   {g:16s} n={len(v):3d}  median {np.median(v):6.2g}  90% {np.percentile(v,90):6.2g}  max {v.max():6.2g}")
    worst = sorted(names, key=lambda n: -e[n])[:15]
    print("   furthest above reference: " + ", ".join(f"{n}({e[n]:.0f}, {cls(n)}, range {float(aud[n]['logspread']):.0f})" for n in worst))
ps.save_panels(figs, f"analysis/catalog_v2/{opts.get('out', 'excess_ratio')}")
