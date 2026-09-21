"""Where in the target does the error sit: the squared residual on the standardized log target
binned in deciles of the target itself, per process, from a run's preds_val.npz
(evaluation.save_predictions). Left: the decile profile (share of the process MSE per decile,
1/10 = flat) of a few named processes. Right: for every process the share of its MSE in the
lowest and highest deciles against the pool's ln|M|^2 range.
    python analysis/catalog_v2/residual_vs_target.py runs/<run> [--show=ee_uu,ee_uugg,...] [--out=name]
Writes analysis/catalog_v2/<out>.{png,pdf} (default residual_vs_target)."""
import csv, glob, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
aud = {re.sub(r"_\d+-\d+GeV_train(_smix)?$", "", r["name"]): r for r in csv.DictReader(open(os.path.join(HERE, "pool_audit.csv"))) if r["role"] == "train"}
f = sorted(glob.glob(os.path.join(args[0], "**", "preds_val.npz"), recursive=True))[-1]
z = np.load(f); pred = z["pred"].reshape(-1); truth = z["truth"].reshape(-1); pid = z["process_id"]; names = list(z["names"])
show = opts.get("show", "ee_mumu,ee_uu,uubar_ttbar_nlo,ee_uug,ee_uugg,ee_wwbb,ee_uubarvevebar,uubar_ttbargg").split(",")
prof = {}; lo_hi = {}
for p, n in enumerate(names):
    m = pid == p
    if m.sum() < 100: continue
    t = truth[m]; r2 = (pred[m] - t) ** 2
    edges = np.quantile(t, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, 9)
    share = np.array([r2[b == k].sum() for k in range(10)]) / r2.sum()
    prof[n] = (share, float(r2.mean()))
    lo_hi[n] = (share[0], share[-1])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
for n in show:
    if n in prof: ax.plot(np.arange(1, 11), prof[n][0], marker="o", ms=3, label=f"{n} (MSE {prof[n][1]:.2g}, range {float(aud[n]['logspread']):.0f})")
ax.axhline(0.1, color="k", lw=0.8, ls="--"); ax.set_xlabel("decile of the standardized target (1 = smallest |M|^2)"); ax.set_ylabel("share of the process's MSE"); ax.set_yscale("log")
ax.set_title("where the error sits, a few processes", fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=7)
ax = axes[1]
ns = [n for n in lo_hi if n in aud and n in NP]
sp = np.array([float(aud[n]["logspread"]) for n in ns]); lo = np.array([lo_hi[n][0] for n in ns]); hi = np.array([lo_hi[n][1] for n in ns])
ax.scatter(sp, lo, s=12, alpha=0.7, label="lowest decile (smallest |M|^2)"); ax.scatter(sp, hi, s=12, alpha=0.7, marker="^", label="highest decile (largest |M|^2)")
ax.axhline(0.1, color="k", lw=0.8, ls="--"); ax.set_yscale("log"); ax.set_xlabel("range of ln|M|^2 in the train pool"); ax.set_ylabel("share of the process's MSE in the decile")
ax.set_title("all processes: the ends of the target against its range", fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=8)
print(f"{len(ns)} processes; median share in the lowest decile {np.median(lo):.2f}, highest {np.median(hi):.2f}; "
      f"processes with > 1/3 of their MSE in the lowest decile: {int((lo > 1/3).sum())}, in the highest: {int((hi > 1/3).sum())}")
for n in sorted(ns, key=lambda n: -lo_hi[n][0])[:10]: print(f"  lowest-decile heavy: {n:24s} low {lo_hi[n][0]:.2f} high {lo_hi[n][1]:.2f} MSE {prof[n][1]:.3g} range {float(aud[n]['logspread']):.0f}")
for n in sorted(ns, key=lambda n: -lo_hi[n][1])[:10]: print(f"  highest-decile heavy: {n:24s} low {lo_hi[n][0]:.2f} high {lo_hi[n][1]:.2f} MSE {prof[n][1]:.3g} range {float(aud[n]['logspread']):.0f}")
fig.tight_layout(); base = os.path.join(HERE, opts.get("out", "residual_vs_target")); fig.savefig(base + ".png", dpi=130); fig.savefig(base + ".pdf"); print("wrote", base + ".{png,pdf}")
