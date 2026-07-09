#!/usr/bin/env python
"""Phase-A data prep: hold out the soft/collinear IR region of ee->uug and build
add-back training sets for the extrapolation study.

Region variable: y_min (master IR resolution var; ->0 in soft & collinear limits),
from ir_observables. Held-out region = y_min < c (c = 15th percentile of y_min, the
deep-IR tail). Split: FAR (y_min>c) always trained; NEAR (y_min<c) split into a fixed
held-out TEST set and an add-back pool.

For each add-back fraction f, write data_ho_f<ff>/ee_uug_91-1000GeV_amplitudes.npy =
FAR ∪ (fraction f of the add-back pool). Keeping the dataset NAME identical (only the
data_path dir changes) so the diagram sidecar still resolves to ee_uug. The fixed
held-out test set is saved for evaluation. CPU only.
"""
import os
import sys
import numpy as np

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, os.path.join(REPO, "analysis/divergences"))
from extract_ir import ir_observables  # noqa

SRC = os.path.join(REPO, "data/ee_uug_91-1000GeV_amplitudes.npy")
DSNAME = "ee_uug_91-1000GeV_amplitudes.npy"
FRACTIONS = [0.0, 0.05, 0.15, 0.5, 1.0]
PCTL = 15.0            # hold out the deepest-IR PCTL% of events
SEED = 42


def main():
    a = np.asarray(np.load(SRC), dtype=np.float64)
    P = (a.shape[1] - 1) // 5
    pdg = a[0, P * 4:P * 5].astype(int)
    obs = ir_observables(a[:, :P * 4].reshape(-1, P, 4), pdg)
    y = obs["y_min"]
    c = np.percentile(y, PCTL)
    near = y < c
    far = ~near
    print(f"N={len(a)} P={P} pdg={list(pdg)}")
    print(f"y_min cut c={c:.3e} ({PCTL}th pctl); NEAR={near.sum()} FAR={far.sum()}")

    rng = np.random.default_rng(SEED)
    near_idx = np.where(near)[0]
    rng.shuffle(near_idx)
    half = len(near_idx) // 2
    test_idx = np.sort(near_idx[:half])          # fixed held-out TEST (never trained)
    addback_pool = near_idx[half:]               # available to add back
    far_idx = np.where(far)[0]
    print(f"held-out TEST (near)={len(test_idx)}  add-back pool={len(addback_pool)}  FAR train={len(far_idx)}")

    # fixed held-out test set for evaluation (raw rows + y_min)
    outdir = os.path.join(REPO, "analysis/divergences")
    np.savez_compressed(os.path.join(outdir, "uug_heldtest.npz"),
                        rows=a[test_idx].astype(np.float32),
                        y_min=y[test_idx], cut=np.array(c), pdg=pdg)
    print(f"saved held-out test -> analysis/divergences/uug_heldtest.npz")

    for f in FRACTIONS:
        n_add = int(round(f * len(addback_pool)))
        add = addback_pool[:n_add]
        train_idx = np.concatenate([far_idx, add])
        rng2 = np.random.default_rng(SEED)
        rng2.shuffle(train_idx)
        tag = f"{int(round(f*100)):03d}"
        ddir = os.path.join(REPO, f"data_ho_f{tag}")
        os.makedirs(ddir, exist_ok=True)
        out = os.path.join(ddir, DSNAME)
        np.save(out, a[train_idx].astype(np.float32))
        print(f"  f={f:.2f}: train={len(train_idx)} (far {len(far_idx)} + add {n_add}) -> data_ho_f{tag}/")


if __name__ == "__main__":
    main()
