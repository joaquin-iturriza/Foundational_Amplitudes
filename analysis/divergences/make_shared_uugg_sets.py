#!/usr/bin/env python
"""The shared validation set of the sigma-driven generation study (plan sigma_generation, step 3). CPU only.

Every arm and seed of l2_online_uugg.py --setup current selects its checkpoint on this one labelled set (--val_path),
so checkpoint choice never moves with the keep rule. It is drawn like the catalog-density arm: candidates from the
loop's proposal (IR-democratic + RAMBO, y_lo 1e-6, lowered cuts), `oversample` per kept event, all labelled, kept
without replacement with the catalog weight (catalog_logweight, f_flat 0.35, 60 bins). Seed 778, disjoint from the
training seeds (1000+s, 2000+97s+r) and from the held-out set (make_heldout_uugg.py, seed 777). Also builds the
process's standalone if it is missing, so the jobs that follow never race to compile it.
Rows: momenta (NP*4), PDG ids (NP), |M|^2, as the loop's round-0 file.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l2_online_uugg as L   # noqa: E402


def build_val(n, oversample, f, n_bins, y_lo, mix_ir, seed):
    rng = np.random.default_rng(seed)
    P = L.propose_momenta(int(round(oversample * n)), y_lo, mix_ir, L.LOW_CUTS, rng)
    me = L.label_events(P)
    logw = L.catalog_logweight(np.log(np.clip(me, 1e-300, None)), f=f, n_bins=n_bins)
    w = np.exp(logw - logw[np.isfinite(logw)].max())
    w[~np.isfinite(w)] = 0.0
    keep = rng.choice(len(P), size=min(n, len(P)), replace=False, p=w / w.sum())
    Pk = P[keep]
    return np.concatenate([Pk.reshape(len(Pk), -1), np.tile(L.PDG.astype(np.float64), (len(Pk), 1)),
                           me[keep].reshape(-1, 1)], axis=1).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", default="uugg", choices=list(L.PROCESSES))
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--oversample", type=float, default=6.0)
    ap.add_argument("--cat_f", type=float, default=0.35)
    ap.add_argument("--cat_bins", type=int, default=60)
    ap.add_argument("--y_lo", type=float, default=1e-6)
    ap.add_argument("--mix_ir", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=778)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    L.set_process(args.process)
    out = args.out or os.path.join(L.REPO, f"data_l2{args.process}/shared/val_catalog_n{args.n}_s{args.seed}.npy")
    if os.path.exists(out):
        print(f"[val] {out} exists, kept", flush=True)
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = build_val(args.n, args.oversample, args.cat_f, args.cat_bins, args.y_lo, args.mix_ir, args.seed)
    np.save(out, rows)
    lm = np.log(rows[:, -1])
    print(f"[val] wrote {out}: {len(rows)} events, log|M|^2 in [{lm.min():.2f}, {lm.max():.2f}]", flush=True)


if __name__ == "__main__":
    main()
