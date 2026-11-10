#!/usr/bin/env python
"""L0 --- the general, process-agnostic coverage resampler: flat in log|M|^2.

Thread D's general default (Sec. sec:div-handoff): every divergence type (soft/collinear
IR, s-channel resonance, threshold) is an *amplitude extreme*, and |M|^2 is the one
coordinate always available (computed to label every event). So the structure-agnostic
coverage criterion is EQUAL TRAINING DENSITY PER AMPLITUDE DECADE --- flat in log|M|^2 ---
which also matches the log-MSE equal-footing loss. No physical binning, no y_min, no sqrt(s):
the resampler reads only the amplitude column.

Given a RAMBO pool (uniform in phase space, hence peaked at the physical measure |M|^2 dPhi
-> extremes starved), we cannot *draw* a target |M|^2, so we shape the density by importance
resampling the existing pool: histogram u = log|amp| into equal-width bins and draw with
probability pi_i = 1 / count(bin(i)), i.e. equal total mass per bin = flat density in u.

Emits TWO training pools of identical size N (fair-compute A/B), differing ONLY in sampling
density, plus a COMMON held-out test split (fixed seed, shared by both arms):
  raw     : uniform subsample of the pool -> native RAMBO density (extremes starved)
  flatlogm: pi ∝ 1/hist(log|amp|)        -> flat per amplitude decade (extremes covered)

Validation is binned by a divergence coordinate the METHOD NEVER USES (e.g. sqrt(s) for the
ee->uu Z resonance), proving the coverage fix is structure-agnostic.
"""
import argparse
import os

import numpy as np


def flat_logm_weights(logamp, bins, eps_frac=None):
    """pi ∝ 1/hist_count over equal-width bins of log|amp| -> flat density per decade.

    Returns (pi, edges, counts). eps_frac (optional) floors per-bin count at
    eps_frac * mean_count so empty/near-empty bins don't get infinite mass.
    """
    counts, edges = np.histogram(logamp, bins=bins)
    which = np.clip(np.digitize(logamp, edges[1:-1]), 0, len(counts) - 1)
    c = counts.astype(float)
    if eps_frac is not None:
        c = np.maximum(c, eps_frac * c[c > 0].mean())
    w = np.where(counts[which] > 0, 1.0 / c[which], 0.0)
    pi = w / w.sum()
    return pi, edges, counts


def _decade_table(logamp10, label, ref=None):
    """Per-decade fraction of log10|M|^2, optionally vs a reference density."""
    lo, hi = np.floor(logamp10.min()), np.ceil(logamp10.max())
    edges = np.arange(lo, hi + 1)
    frac, _ = np.histogram(logamp10, bins=edges)
    frac = 100.0 * frac / frac.sum()
    return edges, frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="input pool .npy (rows: ... , amp in --ampcol)")
    ap.add_argument("--dataset", required=True, help="dataset basename for the output .npy")
    ap.add_argument("--outroot", default=".", help="root under which data_<arm>_<tag>/ dirs are written")
    ap.add_argument("--tag", default="eeuu", help="arm-dir suffix: data_flatlogm_<tag>/, data_raw_<tag>/")
    ap.add_argument("--ampcol", type=int, default=-1, help="amplitude column index (default last)")
    ap.add_argument("--n", type=int, default=400000, help="events per training pool (both arms)")
    ap.add_argument("--bins", type=int, default=40, help="equal-width log|amp| bins for the flat target")
    ap.add_argument("--test_frac", type=float, default=0.1, help="common held-out test fraction")
    ap.add_argument("--eps_frac", type=float, default=None, help="floor per-bin count at eps_frac*mean")
    ap.add_argument("--replace", action="store_true", default=True, help="draw with replacement (default)")
    ap.add_argument("--no_replace", dest="replace", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = np.load(args.pool)
    amp = np.asarray(rows[:, args.ampcol], float)
    assert np.all(amp > 0) or np.all(np.abs(amp) > 0), "amp has zeros; log undefined"
    logamp = np.log(np.abs(amp))
    rng = np.random.RandomState(args.seed)

    # --- common held-out test split (shared by both arms) ---
    perm = rng.permutation(len(rows))
    n_test = int(round(args.test_frac * len(rows)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    tr_rows, tr_log = rows[train_idx], logamp[train_idx]

    # --- flat-in-log|M|^2 proposal over the TRAIN part ---
    pi, edges, counts = flat_logm_weights(tr_log, args.bins, args.eps_frac)

    arms = {
        "raw":      np.full(len(tr_rows), 1.0 / len(tr_rows)),   # native RAMBO density
        "flatlogm": pi,                                          # flat per amplitude decade
    }
    print(f"pool={args.pool}  N_train={len(tr_rows)}  N_test={n_test}  "
          f"log|M|^2 span {tr_log.min():.2f}..{tr_log.max():.2f} nat "
          f"({(tr_log.max()-tr_log.min())/np.log(10):.2f} decades)")
    for name, p in arms.items():
        idx = rng.choice(len(tr_rows), size=args.n, replace=args.replace, p=p)
        new = tr_rows[idx]
        outdir = os.path.join(args.outroot, f"data_{name}_{args.tag}")
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, f"{args.dataset}.npy")
        np.save(out, new)
        uniq = len(np.unique(idx))
        l10 = np.log10(np.abs(np.asarray(new[:, args.ampcol], float)))
        e, fr = _decade_table(l10, name)
        deca = "  ".join(f"[{int(e[i])},{int(e[i+1])}):{fr[i]:.1f}%" for i in range(len(fr)))
        print(f"{name:>9}: {out}  N={len(new)} unique={uniq} ({100*uniq/args.n:.0f}%)")
        print(f"           per-decade log10|M|^2: {deca}")

    # write the common test split once (next to the flatlogm arm; both arms eval on it)
    tdir = os.path.join(args.outroot, f"data_test_{args.tag}")
    os.makedirs(tdir, exist_ok=True)
    tpath = os.path.join(tdir, f"{args.dataset}.npy")
    np.save(tpath, rows[test_idx])
    print(f"     test: {tpath}  N={n_test} (common held-out, native RAMBO density)")


if __name__ == "__main__":
    main()
