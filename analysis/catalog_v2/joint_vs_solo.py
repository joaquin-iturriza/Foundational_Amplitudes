"""Two measurements of the joint-training gap (docs/results.tex, catalog census). Every ratio
is joint over solo, so above one means the joint run is worse.
(1) steps: the joint run's per-process final loss over the solo loss of its multiplicity at
    the same per-process compute, per class, against the horizon. Joint runs runs/steps_t<N>_s*
    (three seeds); solo reference = best trial of sweeps/ref_t<N>_<process> (bs 34, N steps).
(2) interference: each process in the full run (runs/steps_t1000_s*) over the same process in
    its multiplicity subset trained alone at the same per-process compute
    (runs/subset_<2to2|2to3|2to4>_s*).
    python analysis/catalog_v2/joint_vs_solo.py [--steps=1000,2000,4000] [--arm=geo]
--arm=<name> reads the joint runs runs/steps_<name>_t<N>_s* instead (the same curve under another
training aggregation) and writes joint_vs_solo_<name>_*; without it, the arithmetic-mean runs.
Also prints each class's median joint loss per horizon and its slope per doubling of steps.
Writes analysis/catalog_v2/joint_vs_solo_a (steps), _b (ECDF of the ratio), _c (per process)."""
import glob, json, os, re, sys
import numpy as np
from matplotlib.ticker import NullFormatter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import census as C
import plot_style as ps
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
STEPS = [int(x) for x in opts.get("steps", "1000,2000,4000").split(",")]
ARM = opts.get("arm", ""); PRE = f"steps_{ARM}_" if ARM else "steps_"
NP = json.load(open(os.path.join(HERE, "n_particles.json"))); s27, all50 = C.signed_classes()
def cls(n):
    if n in all50: return "signed 1-loop"
    if n.endswith("_nlo") or n.endswith("_loop"): return "positive 1-loop"
    if n in C.NEEDLE or "__mz" in n: return "resonant 2->2"
    return f"tree 2->{NP[n]-2}"
CLASSES = ["tree 2->2", "resonant 2->2", "tree 2->3", "tree 2->4", "positive 1-loop", "signed 1-loop"]
COLS = [ps.C.blue, ps.C.sky, ps.C.vermillion, ps.C.green, ps.C.orange, ps.C.purple]
MULT = ((4, ps.C.blue), (5, ps.C.vermillion), (6, ps.C.green))
REFP = {4: "ee_uu", 5: "ee_uug", 6: "ee_uugg"}
def runs(pat):
    out = []
    for r in sorted(glob.glob(os.path.join(ROOT, pat))):
        d = C.metrics(r); proc = C.at_best(d)[2]; bl = C.best_not_last(d)
        if bl: print(f"  {os.path.relpath(r, ROOT)}: best checkpoint at validation {bl[0] + 1} of {bl[1]}, last/best {bl[2]:.3g}")
        out.append({n: v for n, v in proc.items() if n in NP})
    return out
def solo(N, k):
    f = os.path.join(ROOT, "sweeps", f"ref_t{N}_{REFP[k]}", "summary.txt")
    vals = [float(m.group(1)) for m in re.finditer(r"val_loss=([0-9.eE+-]+)", open(f).read())] if os.path.exists(f) else []
    return min(vals) if vals else float("nan")
(fa, axa), (fb, axb), (fc, axc) = ps.panels(3)
# (1) steps
print("== joint / solo at equal per-process compute, median over the class (mean over seeds ± spread)")
print(f"{'class':16s} " + " | ".join(f"{N:>14d}" for N in STEPS))
curves = {c: [] for c in CLASSES}
absm = {c: [] for c in CLASSES}
for N in STEPS:
    rs = runs(f"runs/{PRE}t{N}_s*"); ref = {k: solo(N, k) for k in REFP}
    for c in CLASSES:
        v = [np.median([r[n] for n in r if cls(n) == c]) for r in rs]
        absm[c].append((np.mean(v), np.std(v, ddof=1)) if len(v) > 1 else (np.mean(v) if v else np.nan, 0.0))
    for c in CLASSES:
        per_seed = [np.median([r[n] / ref[NP[n]] for n in r if cls(n) == c]) for r in rs] if rs else []
        curves[c].append((np.mean(per_seed) if per_seed else np.nan, np.std(per_seed, ddof=1) if len(per_seed) > 1 else 0.0, len(per_seed)))
    print(f"  solo refs at {N}: " + ", ".join(f"2->{k-2} {ref[k]:.3g}" for k in REFP) + f"  ({len(rs)} joint seeds)")
for c in CLASSES:
    print(f"{c:16s} " + " | ".join(f"{m:6.2f} ±{s:4.2f} ({k})" for m, s, k in curves[c]))
print("\n== joint class median (mean over seeds ± spread), and the gain per doubling of steps")
print(f"{'class':16s} " + " | ".join(f"{N:>16d}" for N in STEPS) + " | gain/doubling")
for c in CLASSES:
    m = np.array([x[0] for x in absm[c]])
    g = " ".join(f"{a/b:4.2f}" for a, b in zip(m[:-1], m[1:]))
    print(f"{c:16s} " + " | ".join(f"{a:8.3g} ±{s:6.2g}" for a, s in absm[c]) + f" | {g}")
for c, col in zip(CLASSES, COLS):
    m = np.array([x[0] for x in curves[c]]); s = np.array([x[1] for x in curves[c]])
    axa.errorbar(STEPS, m, yerr=s, marker="o", capsize=3, color=col, label=C.CLASS_LABEL[c])
axa.axhline(1, color=ps.C.grey, ls="--", label="equal to solo")
axa.set_xscale("log"); axa.set_yscale("log"); axa.set_xlabel("training steps")
axa.set_xticks(STEPS, [str(n) for n in STEPS]); axa.xaxis.set_minor_formatter(NullFormatter())
axa.set_ylabel(r"$\mathrm{MSE}_{\rm joint}\,/\,\mathrm{MSE}_{\rm solo}$ (class median)")
ps.shared_legend(fa, axa, ncol=2)   # seven entries do not fit inside the box
# (2) interference
full = [] if ARM else runs("runs/steps_t1000_s*")   # the subset runs are under the arithmetic mean
fullm = {n: np.mean([r[n] for r in full]) for n in full[0]} if full else {}
print("\n== full run / subset alone, same per-process compute (1000 steps), seed means")
for lab, k in (("2to2", 4), ("2to3", 5), ("2to4", 6)):
    rs = runs(f"runs/subset_{lab}_s*")
    if not rs or not fullm: continue
    sub = {n: np.mean([r[n] for r in rs]) for n in rs[0] if n in fullm}
    ratio = {n: fullm[n] / sub[n] for n in sub}
    col = dict(MULT)[k]
    v = np.sort(list(ratio.values()))
    axb.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=col, label=rf"$2\to{k-2}$ ({len(v)} processes)")
    print(f"  2->{k-2}: median full/alone {np.median(v):.2f}, quartiles {np.percentile(v,25):.2f}-{np.percentile(v,75):.2f}; worse in the full run {int((v>1).sum())}, better {int((v<1).sum())}")
    for c in CLASSES:
        vv = [ratio[n] for n in ratio if cls(n) == c]
        if vv: print(f"      {c:16s} n={len(vv):3d} median {np.median(vv):.2f}")
    axc.scatter([sub[n] for n in sub], [fullm[n] for n in sub], color=col, alpha=0.6, label=rf"$2\to{k-2}$")
axb.axvline(1, color=ps.C.grey, ls="--", label="equal")
axb.set_xscale("log"); axb.set_xticks([0.5, 1, 2], ["0.5", "1", "2"]); axb.xaxis.set_minor_formatter(NullFormatter()); axb.set_xlabel(r"$\mathrm{MSE}_{\rm full\ catalog}\,/\,\mathrm{MSE}_{\rm multiplicity\ alone}$")
axb.set_ylabel("fraction of processes")
ps.shared_legend(fb, axb, ncol=2)
lim = [1e-4, 1e1]; axc.plot(lim, lim, color=ps.C.grey, ls="--", label="equal")
axc.set_xscale("log"); axc.set_yscale("log")
axc.set_xlabel(r"MSE, multiplicity trained alone"); axc.set_ylabel(r"MSE, full catalog")
ps.legend(axc, "upper left")
ps.save_panels([fa, fb, fc], "analysis/catalog_v2/joint_vs_solo" + (f"_{ARM}" if ARM else ""))
