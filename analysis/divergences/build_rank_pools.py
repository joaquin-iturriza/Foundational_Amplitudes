#!/usr/bin/env python
"""Q2 rank-vs-magnitude: build MAGNITUDE-CONTROLLED reweighted pools (CPU).

The dense sweep showed pi ∝ Q*sigma uses sigma's MAGNITUDES, not just its rank, so a
Spearman-rho-matched synthetic sigma (scrambled magnitudes) badly under-performs the real
head. To isolate rank from magnitude, every 'rank' arm here uses the SAME weight multiset
-- the true error-magnitude profile Phi = sort(|r|) -- REASSIGNED by a different ranking:
    pi_rank(R) ∝ Q * Phi[ rank(R) ]
so magnitude is held identical across arms and only the ORDERING R varies. Compared with
the magnitude arms (pi ∝ Q*sigma_real, pi ∝ Q*|r|) already trained, this cleanly separates:
  rank_real vs rank_synth046 (same rho, same magnitude) -> is the real-sigma edge rank or magnitude?
  rank_real vs mag_real       -> does sigma's own magnitude beat the true profile by its rank?
  rank_synth{030,046,070}+oracle+baseQ -> the pure-rank tolerance curve (magnitude controlled).
Writes data_reweight_rank_<arm>/.
"""
import argparse
import os
import sys

import numpy as np

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
WT = os.path.join(REPO, "worktrees/wt-heterosc")
sys.path.insert(0, os.path.join(WT, "analysis/divergences"))
from q2_sigma_tolerance import qflat_weights, make_sigma_rho, _spearman  # noqa


def rank_pos(x):
    """0..N-1 integer rank position (ascending), ties broken by argsort order."""
    pos = np.empty(len(x), dtype=np.int64)
    pos[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma_npz", default=os.path.join(WT, "analysis/divergences/train_pool_sigma_antenna.npz"))
    ap.add_argument("--pool", default=os.path.join(REPO, "data_deep_antenna/ee_uug_91-1000GeV_amplitudes.npy"))
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = np.load(args.pool)
    d = np.load(args.sigma_npz)
    err = np.abs(np.asarray(d["pred_logamp"], float) - np.asarray(d["true_logamp"], float))
    sigma = np.asarray(d["sigma_ln"], float)
    y_min = np.asarray(d["y_min"], float)
    Q, _ = qflat_weights(y_min)
    Phi = np.sort(err)                          # true error-magnitude profile (ascending)
    grid = np.linspace(0, 1, 21)

    # ranking sources R (only their ORDER is used)
    rankings = {"rank_real": sigma}
    for r in (0.30, 0.46, 0.70):
        rng = np.random.RandomState(args.seed + int(round(r * 100)))
        s, ra = make_sigma_rho(err, r, rng, grid=grid)
        rankings[f"rank_synth{int(round(r*100)):02d}"] = s
        print(f"synth rho{r:.2f} achieved {ra:.3f}")
    print(f"real sigma rho={_spearman(sigma, err):.3f}")

    rng = np.random.RandomState(args.seed)
    for name, R in rankings.items():
        w = Phi[rank_pos(R)]                     # true-magnitude multiset, reassigned by R's order
        pi = Q * w
        pi = pi / pi.sum()
        idx = rng.choice(len(rows), size=args.n, replace=False, p=pi)
        outdir = os.path.join(REPO, f"data_reweight_{name}")
        os.makedirs(outdir, exist_ok=True)
        np.save(os.path.join(outdir, "ee_uug_91-1000GeV_amplitudes.npy"), rows[idx])
        deep = 100 * np.mean(y_min[idx] < 1e-3)
        print(f"{name:>14}: N={args.n} deepIRfrac={deep:.1f}%  -> {outdir}")


if __name__ == "__main__":
    main()
