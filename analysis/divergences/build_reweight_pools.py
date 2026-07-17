#!/usr/bin/env python
"""Q2 confirmation, step 2: build sigma-reweighted training pools (CPU).

Resample the 400k antenna pool by pi(x) ∝ Q(x) * score(x)^alpha for several score
sources of DIFFERENT rank quality, so the only thing that varies across arms is how
well the reweighting signal orders events by true error. score = a proxy for sqrt(E):

  baseQ   : score = 1            -> pi ∝ Q          (log-flat/decade, NO sigma signal)
  oracle  : score = |r_true|     -> pi ∝ Q*|r|      (perfect signal, rho=1 ceiling)
  sigma   : score = sigma_ln     -> pi ∝ Q*sigma    (real achievable head, rho~0.46)
  deg029  : score = synthetic rho~0.29              (weak within-process quality)

alpha=1 is the variance-optimal proposal pi ∝ Q*sqrt(E)=Q*sigma (adaptive_ir). Each
pool is resampled WITH replacement to the same N=400k (fixed training budget), so the
cross-arm comparison isolates WHERE the budget is spent. Writes
data_reweight_<arm>/ee_uug_91-1000GeV_amplitudes.npy for the finetune array.
"""
import argparse
import os
import sys

import numpy as np

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
WT = os.path.join(REPO, "worktrees/wt-heterosc")
sys.path.insert(0, os.path.join(WT, "analysis/divergences"))
from q2_sigma_tolerance import qflat_weights, make_sigma_rho, _spearman  # noqa


def resample(rows, pi, n, rng, replace=False):
    # WITHOUT replacement to a fixed N: every arm trains on the SAME-size unique set,
    # differing only in WHICH events the signal selects (removes a unique-count confound
    # that with-replacement resampling introduces when pi is more/less peaked per arm).
    idx = rng.choice(len(rows), size=n, replace=replace, p=pi)
    return rows[idx], idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(REPO, "data_deep_antenna/ee_uug_91-1000GeV_amplitudes.npy"))
    ap.add_argument("--sigma_npz", default=os.path.join(WT, "analysis/divergences/train_pool_sigma_antenna.npz"))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=200000)   # < pool -> sample without replacement
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = np.load(args.pool)
    d = np.load(args.sigma_npz)
    err = np.abs(np.asarray(d["pred_logamp"], float) - np.asarray(d["true_logamp"], float))
    sigma = np.asarray(d["sigma_ln"], float)
    y_min = np.asarray(d["y_min"], float)
    assert len(rows) == len(err), (len(rows), len(err))
    rng = np.random.RandomState(args.seed)

    Q, _ = qflat_weights(y_min)
    sig_deg, rho_deg = make_sigma_rho(err, 0.29, rng)
    print(f"real sigma rho={_spearman(sigma, err):.3f}  deg sigma rho={rho_deg:.3f}")

    arms = {
        "baseQ":  np.ones_like(err),
        "oracle": np.clip(err, 1e-12, None),
        "sigma":  np.clip(sigma, 1e-12, None),
        "deg029": np.clip(sig_deg, 1e-12, None),
    }
    for name, score in arms.items():
        pi = Q * score ** args.alpha
        pi = pi / pi.sum()
        new_rows, idx = resample(rows, pi, args.n, rng)
        outdir = os.path.join(REPO, f"data_reweight_{name}")
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, "ee_uug_91-1000GeV_amplitudes.npy")
        np.save(out, new_rows)
        uniq = len(np.unique(idx))
        # per-decade share for a sanity readout
        ly = np.log10(np.clip(y_min[idx], 1e-30, None))
        print(f"{name:>7}: wrote {out}  N={len(new_rows)}  unique={uniq} "
              f"({100*uniq/args.n:.0f}%)  deepIR frac(y<1e-3)={100*np.mean(y_min[idx]<1e-3):.1f}%")


if __name__ == "__main__":
    main()
