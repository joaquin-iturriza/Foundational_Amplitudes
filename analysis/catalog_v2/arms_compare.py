"""The aggregation arms side by side: every process's validation curve, and the final loss by
class (ECDF). Reads plots_0/per_process_metrics.json of each run.
    python analysis/catalog_v2/arms_compare.py "label=runs/<run>" ["label=runs/<run>" ...] [--out=name]
Writes analysis/catalog_v2/<out>_curves_a, _b, ... and <out>_ecdf_a, _b, ... (png+pdf, default
arms_compare), one panel per arm, the arm as the legend title; the combined number and the
per-class medians are printed."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import census as C
import plot_style as ps
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
runs = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in sys.argv[1:] if not a.startswith("--")]
out = opts.get("out", "arms_compare")
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
s27, all50 = C.signed_classes()
def cls(n):
    if n in all50: return "signed 1-loop"
    if n.endswith("_nlo") or n.endswith("_loop"): return "positive 1-loop"
    if n in C.NEEDLE or "__mz" in n: return "resonant 2->2"
    return f"tree 2->{NP[n]-2}"
CLASSES = ["tree 2->2", "resonant 2->2", "tree 2->3", "tree 2->4", "positive 1-loop", "signed 1-loop"]
CCOL = [ps.C.blue, ps.C.sky, ps.C.vermillion, ps.C.green, ps.C.orange, ps.C.purple]
MCOL = {4: ps.C.blue, 5: ps.C.vermillion, 6: ps.C.green}
curves_figs, ecdf_figs = ps.panels(len(runs)), ps.panels(len(runs))
for (fc, axc), (fe, axe), (label, path) in zip(curves_figs, ecdf_figs, runs):
    js = sorted(glob.glob(os.path.join(path, "**", "per_process_metrics.json"), recursive=True))[-1]
    d = json.load(open(js)); every = d["validate_every_n_steps"]
    curves = d["proc_val_losses_no_reg"]; final = {n: v[-1] for n, v in curves.items() if v and n in NP}
    for n, v in curves.items():
        if n in NP:
            # 478 overlapping curves: the thin translucent line is what keeps the density readable
            axc.plot(every * np.arange(1, len(v) + 1), v, color=MCOL[NP[n]], lw=0.4, alpha=0.35)
    axc.set_yscale("log"); axc.set_xlabel("step"); axc.set_ylabel(r"validation MSE($\log|\mathcal{M}|^2$)")
    for k in (4, 5, 6): axc.plot([], [], color=MCOL[k], label=rf"$2\to{k-2}$")   # legend handles
    ps.process_label(axc, label, loc="upper right")
    ps.shared_legend(fc, axc, ncol=3)
    print(f"{label}: combined validation {d['val_loss_no_reg'][-1]:.3g}")
    for c, col in zip(CLASSES, CCOL):
        v = np.sort([final[n] for n in final if cls(n) == c])
        if len(v):
            axe.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=col, label=f"{C.CLASS_LABEL[c]} ({len(v)})")
            print(f"   {c:16s} n={len(v):3d} median {np.median(v):.3g} 90% {np.percentile(v, 90):.3g}")
    axe.set_xscale("log"); axe.set_xlabel(r"final validation MSE($\log|\mathcal{M}|^2$) per process")
    axe.set_ylabel("fraction of processes")
    ps.process_label(axe, label, loc="upper left")
    ps.shared_legend(fe, axe, ncol=2)
# common axes across arms, so the panels compare by eye
for figs, attr in ((curves_figs, "ylim"), (ecdf_figs, "xlim")):
    lims = [getattr(ax, f"get_{attr}")() for _, ax in figs]
    for _, ax in figs: getattr(ax, f"set_{attr}")(min(l[0] for l in lims), max(l[1] for l in lims))
ps.save_panels(curves_figs, f"analysis/catalog_v2/{out}_curves")
ps.save_panels(ecdf_figs, f"analysis/catalog_v2/{out}_ecdf")
