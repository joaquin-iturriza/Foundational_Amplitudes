#!/usr/bin/env python
"""Q2 dense sweep: build reweighted pools across a SINGLE synthetic sigma-quality axis.

The 4-point confirmation couldn't separate quality LEVEL (rho) from quality TYPE (real
head vs synthetic noise), because its low-rho point was synthetic and its rho0.46 point
was the real head. Here every pool uses ONE construction: sigma_rho = true |r| degraded
with random-rank noise to a controlled Spearman rho. Training an arm per rho isolates the
level, so the MSE-vs-rho curve shows the real shape (smooth ramp vs knee). The real head
(rho~0.457) is trained separately (arm 'sigma') and overlaid at plot time.

pi ∝ Q * sigma_rho^alpha, sampled WITHOUT replacement to a fixed N=200k unique (same
budget as the 4-point arms). Writes data_reweight_rho{NN}/ for the finetune array.
"""
import argparse
import os
import sys

import numpy as np

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
WT = os.path.join(REPO, "worktrees/wt-heterosc")
sys.path.insert(0, os.path.join(WT, "analysis/divergences"))
from q2_sigma_tolerance import qflat_weights, make_sigma_rho, _spearman  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma_npz", default=os.path.join(WT, "analysis/divergences/train_pool_sigma_antenna.npz"))
    ap.add_argument("--pool", default=os.path.join(REPO, "data_deep_antenna/ee_uug_91-1000GeV_amplitudes.npy"))
    ap.add_argument("--rhos", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = np.load(args.pool)
    d = np.load(args.sigma_npz)
    err = np.abs(np.asarray(d["pred_logamp"], float) - np.asarray(d["true_logamp"], float))
    y_min = np.asarray(d["y_min"], float)
    Q, _ = qflat_weights(y_min)
    grid = np.linspace(0, 1, 21)   # coarser blend scan -> faster on 400k events

    manifest = []
    for rho in [float(x) for x in args.rhos.split(",")]:
        rng = np.random.RandomState(args.seed + int(round(rho * 100)))  # distinct but reproducible
        s_rho, rho_ach = make_sigma_rho(err, rho, rng, grid=grid)
        pi = Q * np.clip(s_rho, 1e-12, None) ** args.alpha
        pi = pi / pi.sum()
        idx = rng.choice(len(rows), size=args.n, replace=False, p=pi)
        tag = f"rho{int(round(rho*100)):02d}"
        outdir = os.path.join(REPO, f"data_reweight_{tag}")
        os.makedirs(outdir, exist_ok=True)
        np.save(os.path.join(outdir, "ee_uug_91-1000GeV_amplitudes.npy"), rows[idx])
        deep = 100 * np.mean(y_min[idx] < 1e-3)
        manifest.append((tag, rho, rho_ach, deep))
        print(f"{tag}: rho_target={rho:.2f} rho_achieved={rho_ach:.3f} "
              f"deepIRfrac={deep:.1f}%  -> {outdir}")
    tags = ",".join(t for t, *_ in manifest)
    print("\nTAGS=" + tags)


if __name__ == "__main__":
    main()
