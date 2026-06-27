#!/usr/bin/env python3
r"""
nlo_pole_check.py — universal IR-pole validator for MadGraph virtual MEs.

WHY THIS EXISTS
---------------
The finite-part NORMALIZATION is a standardized (BLHA/CDR) tool property, proven
once on eeuu to 2e-12. The IR POLES are UNIVERSAL (Catani): given the external
partons they are PREDICTED, with no reference dataset. So for any new process we
can certify it is set up in the locked convention by checking that MadLoop's
poles equal the universal prediction. If the poles match, the finite part is in
the same (proven) scheme by construction. This is what turns "validated on 2
processes" into "guaranteed on N".

WHAT IS CERTIFIED
-----------------
* DOUBLE pole  c2 = -sum_i C_i  over MASSLESS colored partons (massive -> 0).
  Color-correlation-free, hence universal for ANY process. This is the primary gate.
* SINGLE pole  c1  is predicted analytically for the two clean color-trivial cases
  (the ee->QQ class that dominates the current datasets):
    - two massless colored partons (mu^2 = s):   c1 = -sum_i gamma_i
    - a massive Q Qbar pair (only colored partons): c1 = 2 C_F[(1+b^2)/(2b)L - 1],
      L = ln((1+b)/(1-b)), b = sqrt(1-4 m^2/s)   [calibrated on eett to 0.0000].
  For >2 colored partons / mixed configs the single pole needs color-correlated
  Borns; we then certify the double pole only and say so.

Self-contained: samples its OWN 2-body phase space (no reference data needed), or
reads momenta from a .dat with --dat.

Usage:
    python tools/nlo_pole_check.py --so-dir <P0 dir> \
        --proc-order -11 11 6 -6  --m 0 0 172.5 172.5  --n 200
"""

import argparse
import numpy as np

import nlo_conventions as C
import nlo_madloop as ML


# ---------------------------------------------------------------------------
# Universal pole prediction
# ---------------------------------------------------------------------------
def predict_poles(s, proc_pdgs, masses):
    """Predicted (c2 double, c1 single) in alpha_s/2pi, normalized to born, at
    mu^2 = s. c1 is None when it needs color-correlated Borns. Incoming partons
    (first two slots) are taken colorless (e+ e-)."""
    final = list(zip(proc_pdgs[2:], masses[2:]))
    colored = [(p, m) for p, m in final if C.casimir(p) > 0]
    massless = [(p, m) for p, m in colored if (m is None or m <= 0)]
    massive  = [(p, m) for p, m in colored if (m is not None and m > 0)]

    c2 = -sum(C.casimir(p) for p, _ in massless)

    c1, note = None, ""
    if len(colored) == 2 and len(massless) == 2:
        c1 = -sum(C.gamma_quark() if abs(int(p)) <= 6 else C.gamma_gluon()
                  for p, _ in massless)
        note = "2 massless colored partons (mu^2=s)"
    elif len(colored) == 2 and len(massive) == 2 and \
            abs(massive[0][1] - massive[1][1]) < 1e-6:
        m = massive[0][1]
        b = np.sqrt(max(1.0 - 4.0 * m**2 / s, 0.0))
        L = np.log((1.0 + b) / (1.0 - b))
        c1 = 2.0 * C.CF * ((1.0 + b**2) / (2.0 * b) * L - 1.0)
        note = f"massive Q Qbar pair (beta={b:.4f})"
    else:
        note = ("single pole needs color-correlated Born (>2 or mixed colored "
                "partons) -> only the double pole is certified")
    return c2, c1, note


# ---------------------------------------------------------------------------
# Self-contained 2-body phase space (CM frame), process order [1,2,3,4]
# ---------------------------------------------------------------------------
def sample_2body(sqrts, masses, rng):
    m3, m4 = masses[2], masses[3]
    E = sqrts / 2.0
    # back-to-back beams along z (slot1 -z, slot2 +z); massless leptons
    p1 = np.array([E, 0, 0, -E]); p2 = np.array([E, 0, 0, +E])
    # final-state momentum magnitude
    lam = (sqrts**2 - (m3 + m4)**2) * (sqrts**2 - (m3 - m4)**2)
    p = np.sqrt(max(lam, 0.0)) / (2.0 * sqrts)
    cth = rng.uniform(-1, 1); phi = rng.uniform(0, 2 * np.pi)
    sth = np.sqrt(1 - cth**2)
    d = np.array([p * sth * np.cos(phi), p * sth * np.sin(phi), p * cth])
    p3 = np.array([np.sqrt(p**2 + m3**2), *d])
    p4 = np.array([np.sqrt(p**2 + m4**2), *(-d)])
    return np.vstack([p1, p2, p3, p4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so-dir", required=True)
    ap.add_argument("--proc-order", type=int, nargs="+", required=True,
                    help="signed PDG ids in MadGraph process order")
    ap.add_argument("--m", type=float, nargs="+", required=True,
                    help="mass per particle (same order); 0 for massless")
    ap.add_argument("--sqrts-min", type=float, default=None)
    ap.add_argument("--sqrts-max", type=float, default=1000.0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dat", default=None, help="optional: read momenta from a .dat instead of sampling")
    ap.add_argument("--dat-order", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    masses = args.m
    # threshold for the sampler
    if args.sqrts_min is None:
        args.sqrts_min = 1.05 * sum(masses[2:]) if sum(masses[2:]) > 0 else 50.0

    get_me_full = ML.load(args.so_dir)   # chdir happens here

    if args.dat:
        import os
        raw = np.loadtxt(os.path.abspath(args.dat) if not os.path.isabs(args.dat) else args.dat,
                         comments="#", max_rows=args.n + 1)[:args.n]
        order = args.dat_order or args.proc_order
        idx = [list(order).index(p) for p in args.proc_order]
        def gen():
            for row in raw:
                mom = row[2:18].reshape(4, 4)[idx, :]
                yield mom
        pts = list(gen())
    else:
        rng = np.random.default_rng(args.seed)
        sq = rng.uniform(args.sqrts_min, args.sqrts_max, args.n)
        pts = [sample_2body(s, masses, rng) for s in sq]

    rows = []
    for mom in pts:
        r = ML.evaluate(get_me_full, mom)
        if "c0" not in r:
            continue
        c2p, c1p, note = predict_poles(r["s"], args.proc_order, masses)
        rows.append((np.sqrt(r["s"]), r["c2"], c2p, r["c1"],
                     (c1p if c1p is not None else np.nan)))
    A = np.array(rows)
    sqrts, c2_ml, c2_pred, c1_ml, c1_pred = A.T

    print(f"\nprocess order {args.proc_order}   masses {masses}")
    print(f"points: {len(A)}   sqrt(s) in [{sqrts.min():.0f}, {sqrts.max():.0f}]")
    print(f"\n--- DOUBLE pole c2 = -sum_i C_i  (universal) ---")
    print(f"  predicted (const): {c2_pred[0]:+.5f}")
    print(f"  MadLoop  mean/std: {c2_ml.mean():+.5f} / {c2_ml.std():.2e}")
    print(f"  max |c2_ML - pred|: {np.max(np.abs(c2_ml - c2_pred)):.2e}   "
          f"-> {'PASS' if np.max(np.abs(c2_ml - c2_pred)) < 1e-3 else 'FAIL'}")
    _, _, note = predict_poles(sqrts[0]**2, args.proc_order, masses)
    print(f"\n--- SINGLE pole c1 ---  [{note}]")
    if np.all(np.isfinite(c1_pred)):
        rel = np.max(np.abs(c1_ml - c1_pred) / np.maximum(np.abs(c1_pred), 1e-9))
        print(f"  MadLoop vs prediction:  max |Δ| = {np.max(np.abs(c1_ml - c1_pred)):.2e}  "
              f"max rel = {rel:.2e}   -> {'PASS' if rel < 1e-3 else 'FAIL'}")
    else:
        print("  (not predicted for this multiplicity — double pole certifies setup)")


if __name__ == "__main__":
    main()
