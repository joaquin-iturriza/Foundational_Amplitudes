#!/usr/bin/env python3
"""kit_benchmark 11c: are the one-loop top-pair pools (recipes/catalog_v4_loops_kit.yaml) what the design needs, before a
parent trains on them? Per pool on CC (eval-only, the pools' own rows): events per decade of beta = sqrt(1 - 4 m_t^2/s)
(the threshold pools must reach 1e-5) and of sqrt(s), the share of negative targets, and the slope of the median
log10|V/B| per beta decade (B from the pool's .born.npy sidecar) (Coulomb: V ~ C_F pi alpha_s / (2 beta) x Born, so slope -> -1 as beta -> 0).

    python tools/kit_loop_pool_check.py [--tag probe] --out analysis/kit_benchmark/data/kit_loop_pool_check.json
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SITE = r'''
import glob, json, os, sys
import numpy as np
sys.path.insert(0, ".")
import siteconf
tag = sys.argv[1]
MT = 172.5
out = {}
roots = [siteconf.DATA_DIR, os.path.join(os.environ["SCRATCH"], "amp_data_cache")]   # val/test, train
fs = sorted({f for r in roots for f in glob.glob(os.path.join(r, "v4_*ttx_nlo__*_amplitudes.npy"))
             if tag in os.path.basename(f)})
for f in fs:
    a = np.load(f, mmap_mode="r"); born = np.load(f[:-4] + ".born.npy").astype(float)
    p = np.asarray(a[:, :16], dtype=float).reshape(-1, 4, 4); y = np.asarray(a[:, -1], dtype=float)
    P = p[:, 2] + p[:, 3]; s = P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(1)
    beta = np.sqrt(np.clip(1 - 4 * MT ** 2 / s, 1e-300, None)); sq = np.sqrt(s)
    lb = np.floor(np.log10(beta)).astype(int); ls = np.floor(np.log10(sq) * 2) / 2
    dec = sorted(set(lb.tolist()))
    med = {int(d): float(np.median(np.log10(np.abs(y[lb == d] / born[lb == d]) + 1e-300))) for d in dec if (lb == d).sum() >= 5}
    ks = sorted(med)
    slope = float(np.polyfit(ks, [med[k] for k in ks], 1)[0]) if len(ks) >= 2 else None
    out[os.path.relpath(f)] = {"n": int(len(y)), "beta_min": float(beta.min()), "beta_max": float(beta.max()),
                               "sqrts_min": float(sq.min()), "sqrts_max": float(sq.max()),
                               "per_beta_decade": {str(d): int((lb == d).sum()) for d in dec},
                               "per_half_decade_sqrts": {str(v): int((ls == v).sum()) for v in sorted(set(ls.tolist()))},
                               "median_log10_absVB_per_beta_decade": {str(k): v for k, v in med.items()},
                               "slope_per_beta_decade": slope, "neg_share": float((y < 0).mean()),
                               "finite": bool(np.isfinite(y).all())}
print("LOOPCHECK " + json.dumps(out))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="", help="a substring of the pool file names (probe: the probe's pools)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from analysis.common.readers import site_python
    d = site_python(SITE, [a.tag or "ttx_nlo"], "LOOPCHECK", timeout=900)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(d, open(a.out, "w"), indent=1)
    for k, v in d.items():
        print(os.path.basename(k), v["n"], "beta %.1e..%.2f" % (v["beta_min"], v["beta_max"]),
              "sqrts %.0f..%.0f" % (v["sqrts_min"], v["sqrts_max"]), "slope", v["slope_per_beta_decade"],
              "neg %.2f" % v["neg_share"], "per beta decade", v["per_beta_decade"])
    print("->", a.out)


if __name__ == "__main__":
    main()
