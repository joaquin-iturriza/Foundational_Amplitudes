"""Two measurements of the joint-training gap (docs/results.tex, catalog census).
(1) steps: the joint run's per-process final loss over the solo loss of its multiplicity at
    the same per-process compute, per class, against the horizon. Joint runs runs/steps_t<N>_s*
    (three seeds); solo reference = best trial of sweeps/ref_t<N>_<process> (bs 34, N steps).
(2) interference: each multiplicity subset trained alone at the same per-process compute
    (runs/subset_<2to2|2to3|2to4>_s*) against the same process in the full run (runs/steps_t1000_s*).
    python analysis/catalog_v2/joint_vs_solo.py [--steps=1000,2000,4000]
Writes analysis/catalog_v2/joint_vs_solo.{png,pdf}."""
import glob, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
import census as C
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
STEPS = [int(x) for x in opts.get("steps", "1000,2000,4000").split(",")]
NP = json.load(open(os.path.join(HERE, "n_particles.json"))); s27, all50 = C.signed_classes()
def cls(n):
    if n in all50: return "signed 1-loop"
    if n.endswith("_nlo") or n.endswith("_loop"): return "positive 1-loop"
    if n in C.NEEDLE or "__mz" in n: return "resonant 2->2"
    return f"tree 2->{NP[n]-2}"
CLASSES = ["tree 2->2", "resonant 2->2", "tree 2->3", "tree 2->4", "positive 1-loop", "signed 1-loop"]
REFP = {4: "ee_uu", 5: "ee_uug", 6: "ee_uugg"}
def runs(pat):
    out = []
    for r in sorted(glob.glob(os.path.join(ROOT, pat))):
        js = sorted(glob.glob(os.path.join(r, "**", "per_process_metrics.json"), recursive=True))
        if js:
            d = json.load(open(js[-1])); out.append({n: v[-1] for n, v in d["proc_val_losses_no_reg"].items() if v and n in NP})
    return out
def solo(N, k):
    f = os.path.join(ROOT, "sweeps", f"ref_t{N}_{REFP[k]}", "summary.txt")
    vals = [float(m.group(1)) for m in re.finditer(r"val_loss=([0-9.eE+-]+)", open(f).read())] if os.path.exists(f) else []
    return min(vals) if vals else float("nan")
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
# (1) steps
print("== joint / solo at equal per-process compute, median over the class (mean over seeds ± spread)")
print(f"{'class':16s} " + " | ".join(f"{N:>14d}" for N in STEPS))
curves = {c: [] for c in CLASSES}
for N in STEPS:
    rs = runs(f"runs/steps_t{N}_s*"); ref = {k: solo(N, k) for k in REFP}
    for c in CLASSES:
        per_seed = [np.median([r[n] / ref[NP[n]] for n in r if cls(n) == c]) for r in rs] if rs else []
        curves[c].append((np.mean(per_seed) if per_seed else np.nan, np.std(per_seed, ddof=1) if len(per_seed) > 1 else 0.0, len(per_seed)))
    print(f"  solo refs at {N}: " + ", ".join(f"2->{k-2} {ref[k]:.3g}" for k in REFP) + f"  ({len(rs)} joint seeds)")
for c in CLASSES:
    print(f"{c:16s} " + " | ".join(f"{m:6.2f} ±{s:4.2f} ({k})" for m, s, k in curves[c]))
ax = axes[0]
for c in CLASSES:
    m = np.array([x[0] for x in curves[c]]); s = np.array([x[1] for x in curves[c]])
    ax.errorbar(STEPS, m, yerr=s, marker="o", ms=4, capsize=3, label=c)
ax.axhline(1, color="k", lw=0.8, ls="--"); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("steps (bs 16384, 478 processes)"); ax.set_ylabel("joint loss / solo loss at equal per-process compute (class median)")
ax.set_title("does the gap close with steps?", fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=7)
# (2) interference
full = runs("runs/steps_t1000_s*")
fullm = {n: np.mean([r[n] for r in full]) for n in full[0]} if full else {}
print("\n== subset alone / full run, same per-process compute (1000 steps), seed means")
ax = axes[1]; ax2 = axes[2]
for lab, k in (("2to2", 4), ("2to3", 5), ("2to4", 6)):
    rs = runs(f"runs/subset_{lab}_s*")
    if not rs or not fullm: continue
    sub = {n: np.mean([r[n] for r in rs]) for n in rs[0] if n in fullm}
    ratio = {n: sub[n] / fullm[n] for n in sub}
    v = np.sort(list(ratio.values())); ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", label=f"2->{k-2} alone ({len(v)} processes, {len(rs)} seeds): median {np.median(v):.2f}")
    print(f"  2->{k-2}: median ratio {np.median(v):.2f}, quartiles {np.percentile(v,25):.2f}-{np.percentile(v,75):.2f}; better alone {int((v<1).sum())}, worse {int((v>1).sum())}")
    for c in CLASSES:
        vv = [ratio[n] for n in ratio if cls(n) == c]
        if vv: print(f"      {c:16s} n={len(vv):3d} median {np.median(vv):.2f}")
    ax2.scatter([fullm[n] for n in sub], [sub[n] for n in sub], s=10, alpha=0.6, label=f"2->{k-2}")
ax.axvline(1, color="k", lw=0.8, ls="--"); ax.set_xscale("log"); ax.set_xlabel("loss alone with its class / loss in the full catalog"); ax.set_ylabel("fraction of processes"); ax.set_title("do the classes interfere?", fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=7)
lim = [1e-4, 1e1]; ax2.plot(lim, lim, "k--", lw=0.8); ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_xlabel("loss in the full catalog"); ax2.set_ylabel("loss alone with its class"); ax2.set_title("per process", fontsize=10); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
fig.suptitle("catalog_v2 on the 5k pools, arithmetic mean at its best HPs, target changes on: the joint-training gap", fontsize=11); fig.tight_layout()
base = os.path.join(HERE, "joint_vs_solo"); fig.savefig(base + ".png", dpi=130); fig.savefig(base + ".pdf"); print("wrote", base + ".{png,pdf}")
