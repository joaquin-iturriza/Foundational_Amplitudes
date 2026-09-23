"""Where in the target does the error sit: the squared residual on the standardized log target
binned in deciles of the target itself, per process, from a run's preds_val.npz
(evaluation.save_predictions). Panel a: the decile profile (share of the process MSE per decile,
1/10 = equal) of a few named processes. Panel b: for every process the share of its MSE in the
lowest and highest deciles against the pool's ln|M|^2 range.
    python analysis/catalog_v2/residual_vs_target.py runs/<run> [--show=ee_uu,ee_uugg,...] [--out=name]
Writes analysis/catalog_v2/<out>_a, _b (png+pdf, default residual_vs_target)."""
import csv, glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import plot_style as ps
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
NP = json.load(open(os.path.join(HERE, "n_particles.json")))
aud = {re.sub(r"_\d+-\d+GeV_train(_smix)?$", "", r["name"]): r for r in csv.DictReader(open(os.path.join(HERE, "pool_audit.csv"))) if r["role"] == "train"}
f = sorted(glob.glob(os.path.join(args[0], "**", "preds_val.npz"), recursive=True))[-1]
z = np.load(f); pred = z["pred"].reshape(-1); truth = z["truth"].reshape(-1); pid = z["process_id"]; names = list(z["names"])
show = opts.get("show", "ee_mumu,ee_uu,uubar_ttbar_nlo,ee_uug,ee_uugg,ee_wwbb,ee_uubarvevebar,uubar_ttbargg").split(",")
PRETTY = {"ee_mumu": r"$e^+e^-\to\mu^+\mu^-$", "ee_uu": r"$e^+e^-\to u\bar u$",
          "ee_uug": r"$e^+e^-\to u\bar ug$", "ee_uugg": r"$e^+e^-\to u\bar ugg$",
          "ee_uu_nlo": r"$e^+e^-\to u\bar u$ (one-loop)", "ee_dd_nlo": r"$e^+e^-\to d\bar d$ (one-loop)",
          "uubar_ttbar_nlo": r"$u\bar u\to t\bar t$ (one-loop)", "ee_wwbb": r"$e^+e^-\to W^+W^-b\bar b$",
          "ee_uubarvevebar": r"$e^+e^-\to u\bar u\nu_e\bar\nu_e$", "uubar_ttbargg": r"$u\bar u\to t\bar tgg$",
          "uubar_ZZ": r"$u\bar u\to ZZ$", "ee_ZZ": r"$e^+e^-\to ZZ$"}
prof = {}; lo_hi = {}
for p, n in enumerate(names):
    m = pid == p
    if m.sum() < 100: continue
    t = truth[m]; r2 = (pred[m] - t) ** 2
    edges = np.quantile(t, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, 9)
    share = np.array([r2[b == k].sum() for k in range(10)]) / r2.sum()
    prof[n] = (share, float(r2.mean()))
    lo_hi[n] = (share[0], share[-1])
(fa, axa), (fb, axb) = ps.panels(2)
for n, col in zip([n for n in show if n in prof], ps.CYCLE[:6] + ["black", "#8C564B"]):   # grey is the equal-share line
    axa.plot(np.arange(1, 11), prof[n][0], marker="o", color=col, label=PRETTY.get(n, n.replace("_", " ")))
axa.axhline(0.1, color=ps.C.grey, ls="--", label="equal share")
axa.set_xlabel(r"decile of $\log|\mathcal{M}|^2$ (1 = smallest)"); axa.set_ylabel("share of the process MSE")
axa.set_yscale("log")
ps.shared_legend(fa, axa, ncol=2)
ns = [n for n in lo_hi if n in aud and n in NP]
sp = np.array([float(aud[n]["logspread"]) for n in ns]); lo = np.array([lo_hi[n][0] for n in ns]); hi = np.array([lo_hi[n][1] for n in ns])
axb.scatter(sp, lo, color=ps.C.blue, alpha=0.7, label=r"lowest decile of $|\mathcal{M}|^2$")
axb.scatter(sp, hi, color=ps.C.vermillion, alpha=0.7, marker="^", label=r"highest decile of $|\mathcal{M}|^2$")
axb.axhline(0.1, color=ps.C.grey, ls="--", label="equal share")
axb.set_yscale("log"); axb.set_xlabel(r"range of $\ln|\mathcal{M}|^2$ in the train pool"); axb.set_ylabel("share of the process MSE")
ps.legend(axb, "lower right")
print(f"{len(ns)} processes; median share in the lowest decile {np.median(lo):.2f}, highest {np.median(hi):.2f}; "
      f"processes with > 1/3 of their MSE in the lowest decile: {int((lo > 1/3).sum())}, in the highest: {int((hi > 1/3).sum())}")
for n in sorted(ns, key=lambda n: -lo_hi[n][0])[:10]: print(f"  lowest-decile heavy: {n:24s} low {lo_hi[n][0]:.2f} high {lo_hi[n][1]:.2f} MSE {prof[n][1]:.3g} range {float(aud[n]['logspread']):.0f}")
for n in sorted(ns, key=lambda n: -lo_hi[n][1])[:10]: print(f"  highest-decile heavy: {n:24s} low {lo_hi[n][0]:.2f} high {lo_hi[n][1]:.2f} MSE {prof[n][1]:.3g} range {float(aud[n]['logspread']):.0f}")
ps.save_panels([fa, fb], f"analysis/catalog_v2/{opts.get('out', 'residual_vs_target')}")
