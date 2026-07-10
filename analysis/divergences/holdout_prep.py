#!/usr/bin/env python
"""Phase-A data prep: hold out an IR region of ee->(uug|uugg) and build add-back
training sets for the extrapolation study.

Three region definitions (--region), each holding out the deepest PCTL% tail:
  ymin      : y_min < c            (master IR var; soft AND collinear conflated)  [default]
  soft      : x_g  < c             (pure soft: gluon energy fraction ->0, any angle)
  collinear : y_min < c AND x_g>xg_hard  (pure collinear: small pair invariant at HARD
              gluon, so the soft pole is switched off)
Split: FAR (outside region) always trained; NEAR (in region) split into a fixed held-out
TEST set and an add-back pool. For each fraction f, write <outprefix>_f<tag>/<dsname> =
FAR u (fraction f of the add-back pool); the dataset NAME is kept identical (only the
data_path dir changes) so a diagram sidecar would still resolve. CPU only.
"""
import argparse
import os
import sys
import numpy as np

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, os.path.join(REPO, "analysis/divergences"))
from extract_ir import ir_observables  # noqa


def region_mask(obs, region, pctl, xg_hard):
    """Return (near_mask, cut_value, region_var_name)."""
    y, xg = obs["y_min"], obs["x_gmin"]
    if region == "ymin":
        c = np.percentile(y, pctl)
        return (y < c), c, "y_min"
    if region == "soft":
        c = np.percentile(xg, pctl)
        return (xg < c), c, "x_g"
    if region == "collinear":
        hard = xg > xg_hard                      # gluon non-soft -> isolate the collinear pole
        c = np.percentile(y[hard], pctl)
        return (y < c) & hard, c, "y_min|x_g>%.2f" % xg_hard
    raise ValueError(region)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(REPO, "data/ee_uug_91-1000GeV_amplitudes.npy"))
    ap.add_argument("--dsname", default="ee_uug_91-1000GeV_amplitudes.npy")
    ap.add_argument("--region", default="ymin", choices=["ymin", "soft", "collinear"])
    ap.add_argument("--outprefix", default=os.path.join(REPO, "data_ho"),
                    help="dir prefix; splits go to <outprefix>_f<tag>/")
    ap.add_argument("--testname", default="uug_heldtest.npz")
    ap.add_argument("--pctl", type=float, default=15.0)
    ap.add_argument("--xg-hard", type=float, default=0.5)
    ap.add_argument("--fractions", default="0,0.05,0.15,0.5,1.0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    fractions = [float(x) for x in args.fractions.split(",")]

    a = np.asarray(np.load(args.src), dtype=np.float64)
    P = (a.shape[1] - 1) // 5
    pdg = a[0, P * 4:P * 5].astype(int)
    obs = ir_observables(a[:, :P * 4].reshape(-1, P, 4), pdg)
    near, c, vname = region_mask(obs, args.region, args.pctl, args.xg_hard)
    far = ~near
    print(f"N={len(a)} P={P} pdg={list(pdg)} region={args.region} ({vname})")
    print(f"cut c={c:.3e} ({args.pctl}th pctl); NEAR={int(near.sum())} FAR={int(far.sum())}")

    rng = np.random.default_rng(args.seed)
    near_idx = np.where(near)[0]
    rng.shuffle(near_idx)
    half = len(near_idx) // 2
    test_idx = np.sort(near_idx[:half])          # fixed held-out TEST (never trained)
    addback_pool = near_idx[half:]               # available to add back
    far_idx = np.where(far)[0]
    print(f"held-out TEST={len(test_idx)}  add-back pool={len(addback_pool)}  FAR train={len(far_idx)}")

    outdir = os.path.join(REPO, "analysis/divergences")
    np.savez_compressed(os.path.join(outdir, args.testname),
                        rows=a[test_idx].astype(np.float32),
                        y_min=obs["y_min"][test_idx], x_gmin=obs["x_gmin"][test_idx],
                        cut=np.array(c), region=np.array(args.region), pdg=pdg)
    print(f"saved held-out test -> analysis/divergences/{args.testname}")

    for f in fractions:
        n_add = int(round(f * len(addback_pool)))
        add = addback_pool[:n_add]
        train_idx = np.concatenate([far_idx, add])
        rng2 = np.random.default_rng(args.seed)
        rng2.shuffle(train_idx)
        tag = f"{int(round(f*100)):03d}"
        ddir = f"{args.outprefix}_f{tag}"
        os.makedirs(ddir, exist_ok=True)
        np.save(os.path.join(ddir, args.dsname), a[train_idx].astype(np.float32))
        print(f"  f={f:.2f}: train={len(train_idx)} (far {len(far_idx)} + add {n_add}) -> {os.path.basename(ddir)}/")


if __name__ == "__main__":
    main()
