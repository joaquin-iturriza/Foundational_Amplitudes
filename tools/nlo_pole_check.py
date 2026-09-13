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
  (colourless beams, exactly two coloured legs in the final state; any number of
  colourless final-state particles may accompany them, e.g. ee->qq+gamma/Z/H):
    - two massless colored partons:  c1 = -sum_i gamma_i - sum_i C_i ln(mu^2/s_qq),
      s_qq the pair invariant mass (= s when the pair is the whole final state,
      which is how the mu^2 = s form was calibrated on eeuu)
    - a massive Q Qbar pair: c1 = 2 C_F[(1+b^2)/(2b)L - 1],
      L = ln((1+b)/(1-b)), b = sqrt(1-4 m^2/s_QQ)   [calibrated on eett to 0.0000;
      the eikonal depends only on the pair's relative velocity, hence on s_QQ].
  For >2 colored partons / coloured beams the single pole needs color-correlated
  Borns; we then certify the double pole only and say so.
* MadLoop's return code is kept per point: an exceptional point (hundreds digit 4,
  stability rescue failed) is reported and excluded from the verdict; more than
  10% of them is itself a FAIL.

Self-contained: samples its OWN 2-body phase space (no reference data needed), or
reads momenta from a .dat with --dat.

Usage:
    python tools/nlo_pole_check.py --so-dir <P0 dir> \
        --proc-order -11 11 6 -6  --m 0 0 172.5 172.5  --n 200
"""

import argparse
import numpy as np
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import nlo_conventions as C
import nlo_madloop as ML


# ---------------------------------------------------------------------------
# Universal pole prediction
# ---------------------------------------------------------------------------
def predict_poles(s, proc_pdgs, masses, mom=None):
    """Predicted (c2 double, c1 single) in alpha_s/2pi, normalized to born, at
    mu^2 = s. The double pole sums the Casimirs of ALL massless coloured legs,
    incoming included (quark-initiated processes). c1 is None when it needs
    color-correlated Borns; the two closed forms below assume colourless beams
    and exactly two coloured final-state legs. ``mom`` (slot order, (n,4)) gives
    the pair invariant mass; without it the pair is assumed to carry all of s."""
    legs = list(zip(proc_pdgs, masses))
    final = legs[2:]
    colored = [(p, m) for p, m in legs if C.casimir(p) > 0]
    colored_final = [(p, m) for p, m in final if C.casimir(p) > 0]
    massless = [(p, m) for p, m in colored if (m is None or m <= 0)]
    massive  = [(p, m) for p, m in colored if (m is not None and m > 0)]

    c2 = -sum(C.casimir(p) for p, _ in massless)

    c1, note = None, ""
    if len(colored) != len(colored_final):
        note = "coloured incoming partons -> only the double pole is certified"
        return c2, None, note
    if len(colored) == 2:
        # invariant mass of the coloured pair (its own eikonal scale)
        idx = [i for i, (p, m) in enumerate(legs) if i >= 2 and C.casimir(p) > 0]
        if mom is not None:
            q = np.asarray(mom)[idx[0]] + np.asarray(mom)[idx[1]]
            s_pair = q[0]**2 - q[1]**2 - q[2]**2 - q[3]**2
        else:
            s_pair = s
        extra = len(final) - 2
        tag = f" + {extra} colourless" if extra else ""
    if len(colored) == 2 and len(massless) == 2:
        c1 = -sum(C.gamma_quark() if abs(int(p)) <= 6 else C.gamma_gluon()
                  for p, _ in massless)
        c1 -= sum(C.casimir(p) for p, _ in massless) * np.log(s / s_pair)
        note = f"2 massless colored partons{tag} (mu^2=s, pair mass from the event)"
    elif len(colored) == 2 and len(massive) == 2 and \
            abs(massive[0][1] - massive[1][1]) < 1e-6:
        m = massive[0][1]
        b = np.sqrt(max(1.0 - 4.0 * m**2 / s_pair, 0.0))
        L = np.log((1.0 + b) / (1.0 - b))
        c1 = 2.0 * C.CF * ((1.0 + b**2) / (2.0 * b) * L - 1.0)
        note = f"massive Q Qbar pair{tag} (beta from the pair mass)"
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
        if len(masses) == 4:
            sq = rng.uniform(args.sqrts_min, args.sqrts_max, args.n)
            pts = [sample_2body(s, masses, rng) for s in sq]
        else:
            # n-body (2->3 and up): the pipeline's RAMBO sampler under the fiducial cuts,
            # rows in the process order handed in (beams first). The check only needs s.
            import mg5_pipeline_final as mg
            ev, _ = mg.sample_nbody_phase_space(args.n, args.sqrts_min, args.sqrts_max,
                                                masses[2:], args.proc_order, rng=rng,
                                                cuts=mg.FIDUCIAL_CUTS if mg.FIDUCIAL_CUTS_ENABLED else None)
            pts = [mom for mom, _ in ev]

    rows, rcs, n_noborn = [], [], 0
    for mom in pts:
        r = ML.evaluate(get_me_full, mom)
        if "c0" not in r:
            n_noborn += 1
            continue
        c2p, c1p, note = predict_poles(r["s"], args.proc_order, masses, mom=mom)
        rows.append((np.sqrt(r["s"]), r["c2"], c2p, r["c1"],
                     (c1p if c1p is not None else np.nan)))
        rcs.append(r["rc"])
    A = np.array(rows)
    rcs = np.array(rcs, dtype=int)
    # MadLoop return code: hundreds digit 2 = stable, 3 = rescued (rotation / quad
    # precision), 4 = exceptional (rescue failed; the number is not trustworthy).
    H = rcs // 100
    ok = H != 4
    n_exc, n_resc = int((H == 4).sum()), int((H == 3).sum())
    if len(A) == 0 or ok.sum() == 0:
        print(f"\nno usable points ({len(pts)} sampled, {n_noborn} with born=0, {n_exc} exceptional) -> FAIL")
        return
    sqrts, c2_ml, c2_pred, c1_ml, c1_pred = A.T
    print(f"\nprocess order {args.proc_order}   masses {masses}")
    print(f"points: {len(A)}   sqrt(s) in [{sqrts.min():.0f}, {sqrts.max():.0f}]   "
          f"MadLoop rc: stable {int((H == 2).sum())}  rescued {n_resc}  exceptional {n_exc}"
          + (f"  born=0 {n_noborn}" if n_noborn else ""))
    exc_ok = n_exc <= 0.1 * len(A)
    if not exc_ok:
        print("  more than 10% exceptional points -> FAIL")
    tol = 1e-3   # relative, on the pole coefficients (MadLoop rescues are ~1e-6 accurate)
    dev2 = np.abs(c2_ml - c2_pred)[ok] / np.maximum(np.abs(c2_pred[ok]), 1e-9)
    print("\n--- DOUBLE pole c2 = -sum_i C_i  (universal) ---")
    print(f"  predicted (const): {c2_pred[0]:+.5f}")
    print(f"  MadLoop  mean/std: {c2_ml[ok].mean():+.5f} / {c2_ml[ok].std():.2e}")
    print(f"  max |c2_ML - pred|: {np.max(np.abs(c2_ml - c2_pred)[ok]):.2e}   "
          f"max rel = {dev2.max():.2e}  (95th pct {np.percentile(dev2, 95):.2e})   "
          f"-> {'PASS' if (dev2.max() < tol and exc_ok) else 'FAIL'}")
    for i in np.argsort(-np.where(ok, np.abs(c2_ml - c2_pred), -1.0))[:3]:
        if abs(c2_ml[i] - c2_pred[i]) / max(abs(c2_pred[i]), 1e-9) > 0.1 * tol:
            print(f"    worst: sqrt(s) {sqrts[i]:7.1f}  rc {rcs[i]}  c2 {c2_ml[i]:+.5f}")
    _, _, note = predict_poles(sqrts[0]**2, args.proc_order, masses)
    print(f"\n--- SINGLE pole c1 ---  [{note}]")
    if np.all(np.isfinite(c1_pred)):
        dev1 = np.abs(c1_ml - c1_pred)[ok] / np.maximum(np.abs(c1_pred[ok]), 1e-9)
        print(f"  MadLoop vs prediction:  max |delta| = {np.max(np.abs(c1_ml - c1_pred)[ok]):.2e}  "
              f"max rel = {dev1.max():.2e}  (95th pct {np.percentile(dev1, 95):.2e})   "
              f"-> {'PASS' if dev1.max() < tol else 'FAIL'}")
        for i in np.argsort(-np.where(ok, np.abs(c1_ml - c1_pred), -1.0))[:3]:
            if abs(c1_ml[i] - c1_pred[i]) / max(abs(c1_pred[i]), 1e-9) > 0.1 * tol:
                print(f"    worst: sqrt(s) {sqrts[i]:7.1f}  rc {rcs[i]}  c1 {c1_ml[i]:+.5f}  pred {c1_pred[i]:+.5f}")
    else:
        print("  (not predicted for this multiplicity: the double pole certifies the setup)")


if __name__ == "__main__":
    main()
