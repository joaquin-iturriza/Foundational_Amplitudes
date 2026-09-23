"""Distribution audit of the generated catalog_v2 pools (train + val), run on the login node.

Per pool: sampling mode recorded in the recipe, sqrt(s) coverage of the window, the share
of events within 10 GeV of M_Z and of the pool's own top-of-amplitude region, the
flatness of the log|M|^2 histogram (max/min bin count over the central 98%), and the
standardized log-amplitude tail (fraction beyond 3 sigma, max |z|), i.e. the quantities
behind the 448-run failure taxonomy (resonance needles, plateauing scan families).
Writes analysis/catalog_v2/pool_audit.csv and two figures (png+pdf)."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import datagen
MZ = 91.1876
OUT = os.path.dirname(os.path.abspath(__file__))

def audit(npy, role):
    a = np.load(npy, mmap_mode="r"); ncol = a.shape[1]; n = (ncol - 1) // 5
    P = np.asarray(a[:, :4 * n]).reshape(-1, n, 4)
    amp = np.asarray(a[:, -1])
    rec = json.load(open(npy.replace(".npy", ".recipe.json"))) if os.path.exists(npy.replace(".npy", ".recipe.json")) else {}
    q = P[:, 0] + P[:, 1]; sq = np.sqrt(np.maximum(q[:, 0]**2 - q[:, 1]**2 - q[:, 2]**2 - q[:, 3]**2, 0))
    lo, hi = float(rec.get("sqrts_min", sq.min())), float(rec.get("sqrts_max", sq.max()))
    la = np.log(np.abs(amp[amp != 0]))
    h, _ = np.histogram(la, bins=40, range=np.percentile(la, [1, 99]))
    flat = h.max() / max(h[h > 0].min(), 1)
    z = (la - la.mean()) / max(la.std(), 1e-12)
    top = np.mean(la > np.percentile(la, 99.5))            # sanity (0.5% by construction)
    return dict(name=os.path.basename(npy).replace("_amplitudes.npy", ""), role=role, n=len(amp),
                mode=(rec.get("sampling") or {}).get("mode", "flat") if isinstance(rec.get("sampling"), dict) else rec.get("sampling", "flat"),
                sq_lo=float(sq.min()), sq_hi=float(sq.max()), win_lo=lo, win_hi=hi,
                z_cover=float(np.mean(np.abs(sq - MZ) < 10)) if lo < MZ < hi else np.nan,
                sq_cover10=float(np.mean(np.abs(sq - MZ) < 10)),
                logspread=float(la.max() - la.min()), flatness=float(flat),
                tail3=float(np.mean(np.abs(z) > 3)), zmax=float(np.abs(z).max()),
                neg=int(np.sum(amp < 0)), zero=int(np.sum(amp == 0)))

PLOT_ONLY = "--plot-only" in sys.argv     # redraw the figure from pool_audit.csv, no pool scan
rows = []
for role, d in (() if PLOT_ONLY else (("train", datagen.train_cache_dir()), ("val", datagen.frozen_dir()))):
    files = sorted(glob.glob(os.path.join(d, "*_amplitudes.npy")))
    files = [f for f in files if (role == "train") or ("val" in os.path.basename(f) or "_val_" in os.path.basename(f) or True)]
    for f in files:
        try:
            rows.append(audit(f, role))
        except Exception as e:
            print("ERR", f, e)
import csv
if PLOT_ONLY:
    num = lambda v: float(v) if v not in ("", "nan") else np.nan
    rows = [{k: (v if k in ("name", "role", "mode") else num(v)) for k, v in r.items()}
            for r in csv.DictReader(open(os.path.join(OUT, "pool_audit.csv")))]
else:
    keys = list(rows[0].keys())
    with open(os.path.join(OUT, "pool_audit.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
tr = [r for r in rows if r["role"] == "train"]
print(f"pools: {len(rows)} ({len(tr)} train)")
print("train sampling modes:", {m: sum(1 for r in tr if r['mode'] == m) for m in set(r['mode'] for r in tr)})
zc = [r for r in tr if not np.isnan(r["z_cover"])]
print(f"pools whose window contains M_Z: {len(zc)}; share within 10 GeV of M_Z: "
      f"min {min(r['z_cover'] for r in zc):.3f} median {np.median([r['z_cover'] for r in zc]):.3f} max {max(r['z_cover'] for r in zc):.3f}")
print("  lowest Z coverage:", [(r['name'], round(r['z_cover'], 3)) for r in sorted(zc, key=lambda r: r['z_cover'])[:8]])
print(f"flatness (max/min bin, central 98%): median {np.median([r['flatness'] for r in tr]):.1f}, "
      f"90th pct {np.percentile([r['flatness'] for r in tr], 90):.1f}, worst:",
      [(r['name'], round(r['flatness'])) for r in sorted(tr, key=lambda r: -r['flatness'])[:8]])
print(f"standardized-log tail >3 sigma: median {np.median([r['tail3'] for r in tr]):.4f}; max |z| median {np.median([r['zmax'] for r in tr]):.1f}, worst:",
      [(r['name'], round(r['zmax'], 1)) for r in sorted(tr, key=lambda r: -r['zmax'])[:6]])
print("window coverage gaps (sampled sqrt(s) range narrower than the window by >5%):",
      [(r['name'], round(r['sq_lo']), round(r['sq_hi'])) for r in tr if (r['sq_lo'] - r['win_lo']) > 0.05 * (r['win_hi'] - r['win_lo']) or (r['win_hi'] - r['sq_hi']) > 0.05 * (r['win_hi'] - r['win_lo'])][:10])
print("negative amplitudes (pools, events):", sum(1 for r in tr if r['neg']), sum(r['neg'] for r in tr))
sys.path.insert(0, os.path.dirname(os.path.dirname(OUT)))
import plot_style as ps
(fa, axa), (fb, axb), (fc, axc) = ps.panels(3)
axa.hist([r["z_cover"] for r in zc], bins=30, color=ps.C.blue)
axa.set_xlabel(r"share of events within 10 GeV of $M_Z$"); axa.set_ylabel(r"train pools with $M_Z$ in the window")
axb.hist(np.log10([r["flatness"] for r in tr]), bins=30, color=ps.C.blue)
axb.set_xlabel(r"$\log_{10}$ (max / min bin of the $\log|\mathcal{M}|^2$ histogram)"); axb.set_ylabel("train pools")
axc.hist([r["zmax"] for r in tr], bins=30, color=ps.C.blue)
axc.set_xlabel(r"largest $|z|$ of the standardized $\log|\mathcal{M}|^2$"); axc.set_ylabel("train pools")
ps.save_panels([fa, fb, fc], "analysis/catalog_v2/pool_audit")
