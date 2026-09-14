"""Sign-changing pools of catalog_v2: the standardized target under the unscaled signed
log (sgn(x) log(1+|x|)) and under the scaled one resolved by preprocessing
(sgn(x) log(1+|x|/s), s the data.signedlog_quantile quantile of |x|).
Writes analysis/catalog_v2/signed_pools.csv.   Run: python analysis/catalog_v2/signed_pools.py
"""
import csv, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import datagen
from preprocessing import preprocess_amplitude, resolve_amp_trafos, signedlog_scale

def kurt(v):
    v = v.ravel(); return float(np.mean(v ** 4) / np.mean(v ** 2) ** 2)

rows = []
for npy in sorted(glob.glob(os.path.join(datagen.train_cache_dir(), "*_amplitudes.npy"))):
    a = np.load(npy, mmap_mode="r"); amp = np.asarray(a[:, -1], dtype=np.float64).reshape(-1, 1)
    if float(amp.min()) > 0: continue
    y0, _, _ = preprocess_amplitude(amp, trafos=["signedlog", "standardization"])
    tr = resolve_amp_trafos(["log", "standardization"], amp)
    y1, _, _ = preprocess_amplitude(amp, trafos=tr)
    rows.append(dict(name=os.path.basename(npy).replace("_amplitudes.npy", ""), n=len(amp),
                     neg_frac=float((amp < 0).mean()), med_abs=float(np.median(np.abs(amp))),
                     scale=signedlog_scale(tr[0]), kurt_unscaled=kurt(y0), kurt_scaled=kurt(y1),
                     maxz_unscaled=float(np.abs(y0).max()), maxz_scaled=float(np.abs(y1).max())))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signed_pools.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} sign-changing pools -> {out}")
print(f"{'pool':28s} {'neg':>5s} {'med|x|':>9s} {'s':>9s} {'kurt0':>8s} {'kurt1':>6s} {'maxz0':>6s} {'maxz1':>6s}")
for r in rows:
    print(f"{r['name']:28s} {r['neg_frac']:5.2f} {r['med_abs']:9.2e} {r['scale']:9.2e} "
          f"{r['kurt_unscaled']:8.0f} {r['kurt_scaled']:6.1f} {r['maxz_unscaled']:6.1f} {r['maxz_scaled']:6.1f}")
k1 = [r['kurt_scaled'] for r in rows]; z1 = [r['maxz_scaled'] for r in rows]
print(f"scaled: kurtosis median {np.median(k1):.1f} max {max(k1):.1f}; max|z| median {np.median(z1):.1f} max {max(z1):.1f}")
