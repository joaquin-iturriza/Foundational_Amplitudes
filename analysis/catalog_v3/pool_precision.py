#!/usr/bin/env python
"""Can catalog_v3's pools be stored in float32 instead of float64 (half the disk, 52 -> 26 GB)? (the user, 2026-10-24:
"but yes, check please"). The network reads a float32 cache either way; what float64 pools still serve is the work done
once at load, before the cast: the amplitude itself and the physics factors of the target computed from the momenta.

Over the existing catalog pools in $DATA_DIR (each row: n*4 momenta, n PDG ids, |M|^2; the val split, 1e4 events each):
  range    every pool: the fraction of |M|^2 outside float32's normal range (|x| < 1.18e-38 or > 3.4e38), the fraction
           negative, log10|M|^2 min / max, and whether every PDG id survives the cast exactly
  target   --n-per-layer pools per final-state multiplicity (seed 0) plus every pool with a massive final leg: with the
           momenta cast to float32 and back, the change in log s_hat, in the emission scales (physics_scales.qperp_logs),
           in the smallest log s_ij, in each leg's m^2, and in log|M|^2 with |M|^2 cast; quoted as max and 99.9% |delta|
Prints one PRECISION line per pool and a summary; writes the summary as JSON to --out.
    python analysis/catalog_v3/pool_precision.py [--n-per-layer 7] [--out $SCRATCH/zz_pool_precision.json]
"""
import argparse, glob, json, os, random, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from physics_scales import minkowski_dots, qperp_logs  # noqa: E402

F32_MIN, F32_MAX = np.finfo(np.float32).tiny, np.finfo(np.float32).max


def split(a):
    n = (a.shape[1] - 1) // 5
    return a[:, : 4 * n].reshape(-1, n, 4), a[:, 4 * n: 5 * n], a[:, -1]


def logs(P):
    """the target's load-time quantities from momenta P (N, n, 4), in float64 arithmetic"""
    P = P.astype(np.float64)
    q = P[:, 0] + P[:, 1]
    lsh = np.log(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1))
    D = minkowski_dots(P)                                    # (N, n, n) p_i.p_j
    n = P.shape[1]
    iu = np.triu_indices(n, 1)
    sij = np.abs(2 * D[:, iu[0], iu[1]])
    m2 = np.einsum("nii->ni", D)
    return lsh, qperp_logs(P), np.log(np.maximum(sij.min(1), 1e-300)), m2


def q999(x):
    x = np.abs(x[np.isfinite(x)])
    return (float(x.max()), float(np.quantile(x, 0.999))) if x.size else (0.0, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-layer", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(os.environ.get("SCRATCH", "/tmp"), "zz_pool_precision.json"))
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(os.environ["DATA_DIR"], "*_val*_amplitudes.npy")))
    rows, by_n = [], {}
    for f in files:
        try:
            x = np.load(f, mmap_mode="r")
        except Exception:
            continue
        if x.ndim != 2 or (x.shape[1] - 1) % 5:
            continue
        P, ids, amp = split(np.asarray(x))
        if not np.all(np.isfinite(amp)) or not len(amp):
            continue
        ab = np.abs(amp)
        r = dict(pool=os.path.basename(f), n=int(P.shape[1]), events=int(len(amp)),
                 out_of_f32=float(np.mean((ab < F32_MIN) | (ab > F32_MAX))), negative=float(np.mean(amp < 0)),
                 log10_min=float(np.log10(ab[ab > 0].min())) if (ab > 0).any() else None,
                 log10_max=float(np.log10(ab.max())) if (ab > 0).any() else None,
                 ids_exact=bool(np.array_equal(ids.astype(np.float32).astype(np.float64), ids)),
                 massive=bool(np.any(np.abs(ids[0, 2:]) == 6) or np.any(np.abs(ids[0, 2:]) == 5)
                              or np.any(np.isin(np.abs(ids[0, 2:]), (23, 24, 25)))))
        rows.append(r)
        by_n.setdefault(r["n"], []).append((f, r))
    rng = random.Random(0)
    pick = {f for n, fr in sorted(by_n.items()) for f, _ in rng.sample(fr, min(len(fr), a.n_per_layer))}
    pick |= {f for fr in by_n.values() for f, r in fr if r["massive"] and r["n"] >= 5 and rng.random() < 0.1}
    for f, r in ((f, r) for fr in by_n.values() for f, r in fr if f in pick):
        P, ids, amp = split(np.asarray(np.load(f)))
        keep = amp != 0
        P, amp = P[keep], amp[keep]
        a64, a32 = logs(P), logs(P.astype(np.float32))
        la = np.log(np.abs(amp))
        r.update(d_log_shat=q999(a32[0] - a64[0]), d_qperp=q999(a32[1] - a64[1]), d_log_smin=q999(a32[2] - a64[2]),
                 d_m2_rel=q999((a32[3] - a64[3]) / np.maximum(np.abs(a64[3]), 1.0)),
                 d_log_amp=q999(np.log(np.abs(amp.astype(np.float32).astype(np.float64))) - la))
    for r in rows:
        if "d_qperp" in r:
            print("PRECISION", json.dumps(r))
    neg = [r for r in rows if r["negative"] > 0]
    summ = dict(pools=len(rows), events=sum(r["events"] for r in rows),
                pools_with_out_of_f32=sum(r["out_of_f32"] > 0 for r in rows),
                worst_out_of_f32=max(rows, key=lambda r: r["out_of_f32"])["pool"] if rows else None,
                worst_out_of_f32_frac=max((r["out_of_f32"] for r in rows), default=0.0),
                log10_amp_min=min((r["log10_min"] for r in rows if r["log10_min"] is not None), default=None),
                log10_amp_max=max((r["log10_max"] for r in rows if r["log10_max"] is not None), default=None),
                pools_with_negative=len(neg), ids_exact=all(r["ids_exact"] for r in rows),
                checked=sum("d_qperp" in r for r in rows))
    for k in ("d_log_shat", "d_qperp", "d_log_smin", "d_m2_rel", "d_log_amp"):
        v = [r[k] for r in rows if k in r]
        summ[k + "_max"] = max(x[0] for x in v) if v else None
        summ[k + "_q999_max"] = max(x[1] for x in v) if v else None
    print("SUMMARY", json.dumps(summ))
    json.dump({"summary": summ, "pools": rows}, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
