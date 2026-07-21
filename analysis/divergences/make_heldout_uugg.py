#!/usr/bin/env python
"""Generate a FIXED held-out deep-IR ee->uugg evaluation set for the L2 comparison.

Both arms (sigma-driven online generation, and any existing baseline) are scored on THIS set, on raw
log|M|^2 (each model's own preprocessing inverted) so training-side preprocessing differences never
bias the comparison. Drawn from the SAME process-agnostic base proposal used in training (IR-democratic
+ RAMBO bulk) so it spans both the O(1) bulk AND the soft/collinear corners; y_min / x_gmin are stored
so error can be reported per IR-resolution decade (where the divergences live). Never trained on
(separate seed). CPU only.
"""
import argparse
import os
import sys

import numpy as np

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WT, "analysis/divergences"))
import l2_online_uugg as L   # propose_momenta / label_events / PDG   # noqa: E402


def ir_observables(P):
    """y_min (min gluon-involving pair invariant / s) and x_gmin (softest gluon energy frac).
    Process-agnostic: gluon/coloured indices come from the l2_online_uugg registry (L.GLUONS/COLORED),
    so this works for uug (1 gluon), uugg (2), uuggg (3)."""
    def dot(a, b): return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)
    Q = P[:, 0] + P[:, 1]; s = dot(Q, Q)
    glu = L.GLUONS; colored = L.COLORED; gset = set(glu)
    pairs = [(colored[a], colored[b]) for a in range(len(colored)) for b in range(a + 1, len(colored))
             if (colored[a] in gset or colored[b] in gset)]
    yv = np.stack([dot(P[:, i] + P[:, j], P[:, i] + P[:, j]) / s for i, j in pairs], 1)
    y_min = np.clip(yv, 1e-14, None).min(1)
    xg = np.stack([2.0 * dot(P[:, g], Q) / s for g in glu], 1)
    return y_min, xg.min(1), 2.0 * P[:, 0, 0]   # y_min, x_gmin, sqrt_s (CM: 2*E_beam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", default="uugg", choices=list(L.PROCESSES),
                    help="uug (Z-res+IR multi-scale) | uugg | uuggg")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--y_lo", type=float, default=1e-8, help="deeper than training to stress the tail")
    ap.add_argument("--mix_ir", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=777, help="disjoint from training seeds")
    ap.add_argument("--peak_frac", type=float, default=0.0,
                    help="uug multi-scale: add this fraction of extra events with sqrt_s near M_Z "
                         "(enriches the Z-peak for a well-measured resonance-region MSE)")
    ap.add_argument("--peak_hi", type=float, default=96.0, help="upper sqrt_s for the peak-enrichment batch")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    L.set_process(args.process)
    if args.out is None:
        args.out = os.path.join(L.REPO, f"analysis/divergences/heldout_{args.process}_deepIR.npz")

    rng = np.random.default_rng(args.seed)
    print(f"[heldout] proposing {args.n} events (y_lo={args.y_lo}, mix_ir={args.mix_ir})", flush=True)
    P = L.propose_momenta(args.n, args.y_lo, args.mix_ir, L.LOW_CUTS, rng)
    # Multi-scale (uug): M_Z=91.19 sits at the very bottom of the sqrt(s)=[91,1000] window, so uniform
    # sampling starves the Z-peak (~0.3%). Enrich it with a dedicated peak-window batch so the
    # resonance-region MSE (and the deep-IR x on-peak split) is well-estimated. Evaluation only --
    # per-region MSEs are unaffected by cross-region proportions; training keeps the uniform base.
    if args.peak_frac > 0:
        n_peak = int(round(args.peak_frac * args.n))
        print(f"[heldout] enriching {n_peak} events near M_Z (sqrt_s in [91,{args.peak_hi}])", flush=True)
        P_pk = L.propose_momenta(n_peak, args.y_lo, args.mix_ir, L.LOW_CUTS, rng,
                                 sqrt_s_lo=91.0, sqrt_s_hi=args.peak_hi)
        P = np.concatenate([P, P_pk], axis=0)
    me2 = L.label_events(P)
    y_min, x_gmin, sqrt_s = ir_observables(P)
    rows = np.concatenate([P.reshape(len(P), -1), np.tile(L.PDG.astype(np.float64), (len(P), 1)),
                           me2.reshape(-1, 1)], axis=1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, rows=rows.astype(np.float64), y_min=y_min, x_gmin=x_gmin,
                        sqrt_s=sqrt_s, pdg=L.PDG)
    print(f"[heldout] saved {args.out}  N={len(rows)}  |M|^2 [{me2.min():.2e},{me2.max():.2e}]", flush=True)
    for lo, hi in [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.01)]:
        n = int(((y_min >= lo) & (y_min < hi)).sum())
        print(f"    y_min[{lo:.0e},{hi:.0e}): {n:6d}  ({100*n/len(rows):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
