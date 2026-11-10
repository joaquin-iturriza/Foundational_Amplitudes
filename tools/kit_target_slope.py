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
FLAT lines: per stratum, the log10-MSE (KIT's, of log10 |M|^2) of predictors that learned nothing about the limit:
each target clipped to its bulk range (_edge) or held at its bulk median (_median).
FEAT lines: the off-shellness input the model reads for a massless parton propagator, o = min over final-state pairs
with a gluon of 2 p_i.p_j / s_hat (data.dimensionless_inputs), as sign(o) log1p(o / eps) (experiment.py) at
eps = data.dimless_offshell_eps 1e-6 (the default) and 1e-12: per stratum its median and the share outside its bulk
[0.5%, 99.5%] range, and per family its slope against log10 singular_var (ln 10 = 2.303 when the feature is
logarithmic across the bins, 0 when log1p has flattened it).
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import kit_hardps as kh  # noqa: E402
from physics_scales import minkowski_dots, qperp_logs  # noqa: E402


def both(rows, n_part):
    mom = rows[:, :4 * n_part].reshape(-1, n_part, 4)
    lnm = np.log(rows[:, -1])
    return lnm / np.log(10.0), (lnm + qperp_logs(mom)) / np.log(10.0)


EPS = (1e-6, 1e-12)


def feats(rows, n_part):
    """{eps: log1p(o / eps)} of o = min over final-state pairs with a gluon of 2 p_i.p_j / s_hat (None: no gluon)"""
    mom = rows[:, :4 * n_part].reshape(-1, n_part, 4)
    pdg = rows[0, 4 * n_part:5 * n_part].astype(int)
    G = minkowski_dots(mom)
    shat = np.abs(G[:, 0, 0] + G[:, 1, 1] + 2 * G[:, 0, 1])
    pairs = [(i, j) for i in range(2, n_part) for j in range(i + 1, n_part) if 21 in (pdg[i], pdg[j])]
    if not pairs:
        return None
    o = np.min(np.stack([2 * np.abs(G[:, i, j]) for i, j in pairs], 1), 1) / shat
    return {e: np.log1p(o / e) for e in EPS}


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
    bf = feats(bulk, n_part)
    flo = {e: np.quantile(v, 0.005) for e, v in bf.items()} if bf else {}
    fhi = {e: np.quantile(v, 0.995) for e, v in bf.items()} if bf else {}
    if bf:
        print("BULKFEAT " + " ".join(f"eps {e:g} [{flo[e]:.3f}, {fhi[e]:.3f}]" for e in EPS))
    print(f"BULK n={len(bp)} plain [{lo['plain']:.3f}, {hi['plain']:.3f}] T [{lo['T']:.3f}, {hi['T']:.3f}] (log10)")
    fits = {}
    for st in kh.strata(a.file, "hard_test"):
        with h5py.File(kh.h5_path(a.file), "r") as f:
            g = f[f"hard_test/{st}"]
            sv = np.asarray(g["singular_var"][:], np.float64) if "singular_var" in g else None
        rows = kh.stratum_rows(a.file, a.target, "hard_test", st)
        p, t = both(rows, n_part)
        hf = feats(rows, n_part) if bf else None
        if hf:
            print(f"FEAT {st} " + " ".join(f"eps {e:g} median {np.median(v):.3f} outside "
                                            f"{np.mean((v < flo[e]) | (v > fhi[e])):.3f}" for e, v in hf.items()))
        out = {k: float(np.mean((v < lo[k]) | (v > hi[k]))) for k, v in (("plain", p), ("T", t))}
        print(f"RANGE {st} n={len(p)} outside bulk range: plain {out['plain']:.3f} T {out['T']:.3f}; "
              f"median plain {np.median(p):.3f} T {np.median(t):.3f}")
        # what a network that learned nothing about the limit scores in KIT's log10-MSE of |M|^2: each target held at
        # the edge of its bulk range (the value clipped to it), and held at its bulk median; T's predictions are
        # mapped back to log10 |M|^2 with the event's own emission scales (p - t is that event's -sum ln q^2 / ln 10)
        flat = {"plain_edge": np.clip(p, lo["plain"], hi["plain"]), "plain_median": np.full_like(p, np.median(bp)),
                "T_edge": p + (np.clip(t, lo["T"], hi["T"]) - t), "T_median": p + (np.median(bt) - t)}
        print(f"FLAT {st} log10-MSE " + " ".join(f"{k} {np.mean((v - p) ** 2):.3e}" for k, v in flat.items()))
        if sv is not None:
            fits.setdefault(st.rsplit("_", 2)[0], []).append((np.log10(sv), p, t, hf))
    for fam, parts in fits.items():
        x = np.concatenate([q[0] for q in parts])
        sp = np.polyfit(x, np.concatenate([q[1] for q in parts]), 1)[0]
        stt = np.polyfit(x, np.concatenate([q[2] for q in parts]), 1)[0]
        print(f"SLOPE {fam} plain {sp:+.3f} T {stt:+.3f} (d log10 / d log10 singular_var, {len(x)} events)")
        if parts[0][3]:
            print(f"FEATSLOPE {fam} " + " ".join(
                f"eps {e:g} {np.polyfit(x, np.concatenate([q[3][e] for q in parts]), 1)[0]:+.3f}" for e in EPS)
                + " (d log1p(o/eps) / d log10 singular_var; ln 10 = 2.303 fully logarithmic)")


if __name__ == "__main__":
    main()
