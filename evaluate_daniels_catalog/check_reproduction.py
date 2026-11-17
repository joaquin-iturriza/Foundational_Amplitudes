#!/usr/bin/env python3
"""The evaluator's scorer path against a stored zero-shot read of the same val pool, event by event.

    python evaluate_daniels_catalog/check_reproduction.py --pool <dc_X_13000-13000GeV_val_lhe_amplitudes.npy> \
        --stored <read.npz> [--seeds 42 1 2 ...] [--n-val 10000] [--n 2000]

--pool: the val pool the stored read scored, with its .recipe.json and .alphas.npy beside it. --stored: the read's
  .npz (<probe>__t, <probe>__q, as collect_heldout_preds.py packs them).
The stored reads drew the Lorentz augmentation from the global torch RNG (before keyed_aug, 2026-11-06), a draw that
cannot be replayed; the evaluator draws it per event from (seed, event). So the check is against the spread of the
augmentation: per event, the stored q (log(|M|^2 F)) against the mean and standard deviation of the evaluator's q over
--seeds, events matched on t; and the stored MSE of log|M|^2 against the evaluator's MSE per seed.
"""
import argparse, os, re, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--stored", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4, 5, 6, 7])
ap.add_argument("--n-val", type=int, default=10000)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--model", default=os.path.join(evaluate.HERE, "model"))
ap.add_argument("--out", default=None, help="save t, the stored q and the evaluator's q per seed (.npz)")
a = ap.parse_args()

name = re.match(r"(dc_.+?)_\d+-\d+GeV_val", os.path.basename(a.pool)).group(1)
st = np.load(a.stored)
t0, q0 = st[name + "__t"].astype(np.float64), st[name + "__q"].astype(np.float64)
i0 = np.argsort(t0, kind="stable")
t0, q0 = t0[i0], q0[i0]
Q = []
for seed in a.seeds:
    z = evaluate.score(name, os.path.abspath(a.pool), a.n_val, a.n, a.model, seed=seed)
    t1, q1 = z["t"].astype(np.float64), z["q"].astype(np.float64)
    i1 = np.argsort(t1, kind="stable")
    assert len(t0) == len(t1) and np.abs(t0 - t1[i1]).max() < 1e-5, "the two reads did not score the same events"
    Q.append(q1[i1])
Q = np.array(Q)
mse = ((Q - t0) ** 2).mean(axis=1)
m, sd = Q.mean(axis=0), Q.std(axis=0, ddof=1)
pull = (q0 - m) / (sd * np.sqrt(1 + 1 / len(Q)))     # a new draw against the mean of K draws
print(f"{name}: {len(t0)} events matched on t, {len(Q)} seeds {a.seeds}")
print(f"  evaluator MSE of log|M|^2 per seed: {' '.join(f'{x:.4f}' for x in mse)} (mean {mse.mean():.4f}, "
      f"sd {mse.std(ddof=1):.4f}); stored read {np.mean((q0 - t0) ** 2):.4f}")
print(f"  per-event sd of q over the seeds: median {np.median(sd):.4f}, rms {np.sqrt(np.mean(sd ** 2)):.4f}")
print(f"  stored q - mean q: rms {np.sqrt(np.mean((q0 - m) ** 2)):.4f}; pull (stored - mean)/sd: mean "
      f"{pull.mean():.3f}, sd {pull.std():.3f}, |pull|<1 {np.mean(np.abs(pull) < 1):.3f}, |pull|<2 "
      f"{np.mean(np.abs(pull) < 2):.3f}")
print(f"  MSE of the mean over seeds: {np.mean((m - t0) ** 2):.4f}")
if a.out:
    np.savez(a.out, t=t0, q_stored=q0, q=Q, seeds=np.array(a.seeds))
