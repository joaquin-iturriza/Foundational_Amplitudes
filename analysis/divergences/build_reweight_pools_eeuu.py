#!/usr/bin/env python
"""L1 step 3 for ee->uu: sigma-reweight-subsample the large mix025 candidate pool down to the
training budget. From ONE big generated pool (data_l1pool_eeuu, mix025-distributed), draw N=400k
WITHOUT replacement under three selection signals, so every arm is the SAME density base emphasized
differently (isolates the reweighting signal; no coverage ceiling since the pool is 1.5M fresh):
  base   : pi = const            -> the L0 mix025 coverage base (control)
  sigma  : pi ∝ sigma^alpha      -> the real head's uncertainty emphasis (the L1 test)
  oracle : pi ∝ |pred-true|^alpha-> perfect-signal ceiling
alpha=1 is the variance-optimal proposal pi ∝ sqrt(E) ~ sigma (adaptive_ir). sqrt(s) is used only
for the diagnostic readout, never to select. Writes data_l1<arm>_eeuu/ for the finetune array."""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
    ap.add_argument("--pool", default=os.path.join(REPO, "data_l1pool_eeuu/ee_uu_91-1000GeV_amplitudes.npy"))
    ap.add_argument("--sigma_npz", default=os.path.join(REPO, "analysis/divergences/l1pool_sigma_eeuu.npz"))
    ap.add_argument("--n", type=int, default=400000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outsuffix", default="", help="appended to arm dir name, e.g. _s1 (seed study)")
    args = ap.parse_args()

    rows = np.load(args.pool)
    d = np.load(args.sigma_npz)
    sigma = np.asarray(d["sigma_ln"], float)
    err = np.abs(np.asarray(d["pred_logamp"], float) - np.asarray(d["true_logamp"], float))
    sqrt_s = np.asarray(d["sqrt_s"], float)
    assert len(rows) == len(sigma), (len(rows), len(sigma))
    rng = np.random.RandomState(args.seed)

    arms = {
        "l1base":   np.ones_like(sigma),
        "l1sigma":  np.clip(sigma, 1e-12, None),
        "l1oracle": np.clip(err, 1e-12, None),
    }
    reg = [(88, 95), (95, 150), (150, 400), (400, 1000)]
    print(f"pool N={len(rows)}  target N={args.n}  alpha={args.alpha}")
    for name, score in arms.items():
        pi = score ** args.alpha
        pi = pi / pi.sum()
        idx = rng.choice(len(rows), size=args.n, replace=False, p=pi)
        outdir = os.path.join(REPO, f"data_{name}{args.outsuffix}_eeuu")
        os.makedirs(outdir, exist_ok=True)
        np.save(os.path.join(outdir, "ee_uu_91-1000GeV_amplitudes.npy"), rows[idx])
        frac = "  ".join(f"[{lo},{hi}):{100*np.mean((sqrt_s[idx]>=lo)&(sqrt_s[idx]<hi)):.1f}%" for lo, hi in reg)
        print(f"{name:>9}: N={args.n} unique={len(np.unique(idx))} | sqrt(s) sel {frac}")


if __name__ == "__main__":
    main()
