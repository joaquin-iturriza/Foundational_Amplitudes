#!/usr/bin/env python3
"""How the emission-scale target of the catalog parent moves into the KIT hard bins, against the plain ln|M|^2.

    python tools/kit_target_slope.py <file> <target> [--n-bulk 200000]

On the truth only (no network): for every hard_test family with a singular variable (soft_x, coll_y, threshold_beta,
... as KIT's src/evaluate.py groups them), the least-squares slope against log10 singular_var of
  plain  log10 |M|^2                               (KIT's 'truth' exponent)
  T      (ln |M|^2 + sum_k ln q_k^2) / ln 10       (data.target_physics_dim_ref=qperp: physics_scales.qperp_logs, the
                                                   emission scales the parent's target divides out; s0, the coupling
                                                   orders and the fixed alpha_s are constants here and drop out)
and, per stratum, the share of its events whose value lies outside the [0.5%, 99.5%] range of the same quantity on
the first n_bulk events of bulk_train: how far each target has to extrapolate past what training saw.
Rows are tools/kit_hardps.py's (catalog leg order, the stripped target). Prints one SLOPE line per family and one
RANGE line per stratum.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import kit_hardps as kh  # noqa: E402
from physics_scales import qperp_logs  # noqa: E402


def both(rows, n_part):
    mom = rows[:, :4 * n_part].reshape(-1, n_part, 4)
    lnm = np.log(rows[:, -1])
    return lnm / np.log(10.0), (lnm + qperp_logs(mom)) / np.log(10.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file")
    ap.add_argument("target")
    ap.add_argument("--n-bulk", type=int, default=200000)
    a = ap.parse_args()
    import h5py
    meta = kh.metadata(a.file)
    n_part = len(meta["particles"])
    with h5py.File(kh.h5_path(a.file), "r") as f:
        g = f["bulk_train/bulk"]
        sl = slice(0, min(a.n_bulk, g["momenta"].shape[0]))
        bulk = kh.group_rows({k: g[k][sl] for k in ("momenta", a.target)}, a.target, meta)
    bp, bt = both(bulk, n_part)
    lo = {"plain": np.quantile(bp, 0.005), "T": np.quantile(bt, 0.005)}
    hi = {"plain": np.quantile(bp, 0.995), "T": np.quantile(bt, 0.995)}
    print(f"BULK n={len(bp)} plain [{lo['plain']:.3f}, {hi['plain']:.3f}] T [{lo['T']:.3f}, {hi['T']:.3f}] (log10)")
    fits = {}
    for st in kh.strata(a.file, "hard_test"):
        with h5py.File(kh.h5_path(a.file), "r") as f:
            g = f[f"hard_test/{st}"]
            sv = np.asarray(g["singular_var"][:], np.float64) if "singular_var" in g else None
        p, t = both(kh.stratum_rows(a.file, a.target, "hard_test", st), n_part)
        out = {k: float(np.mean((v < lo[k]) | (v > hi[k]))) for k, v in (("plain", p), ("T", t))}
        print(f"RANGE {st} n={len(p)} outside bulk range: plain {out['plain']:.3f} T {out['T']:.3f}; "
              f"median plain {np.median(p):.3f} T {np.median(t):.3f}")
        if sv is not None:
            fits.setdefault(st.rsplit("_", 2)[0], []).append((np.log10(sv), p, t))
    for fam, parts in fits.items():
        x = np.concatenate([q[0] for q in parts])
        sp = np.polyfit(x, np.concatenate([q[1] for q in parts]), 1)[0]
        stt = np.polyfit(x, np.concatenate([q[2] for q in parts]), 1)[0]
        print(f"SLOPE {fam} plain {sp:+.3f} T {stt:+.3f} (d log10 / d log10 singular_var, {len(x)} events)")


if __name__ == "__main__":
    main()
