#!/usr/bin/env python
"""Deeper + cleverer phase-space sampling of the ee->uug IR singularity.

Baseline generation is FLAT RAMBO (uniform in Lorentz-invariant phase space); the
soft/collinear region (small y_min) is therefore starved (the 1M production set has
~5 events below y_min=1e-5). This script builds two matched training sets over the
SAME (lowered-cut) fiducial region so the ONLY difference is sampling density:

  uniform : flat RAMBO (reuses mg5_pipeline_final.sample_nbody_phase_space).
  antenna : importance-sample the LO antenna structure dN ∝ da/a · db/b, where
            a = 1-x_u = y(ubar,g), b = 1-x_ubar = y(u,g) are the two gluon-involving
            invariants (x_i = 2E_i/√s). Then y_min = min(a,b) and the density is
            ∝ 1/(a b) (the factorized soft×collinear antenna) — flat per decade in
            each invariant, so events fill every decade down to y_lo instead of
            piling up at O(1). Reaches y_min < 1e-6 at fixed event count.

Labels are the EXACT tree |M|^2 from the compiled C++ standalone (fixed α_s=0.118,
no per-event rescale — verified bit-identical to the production dataset), so the
antenna set's amplitudes are on the same convention as the uniform baseline. We
train on log|M|^2 pointwise, so importance sampling needs NO target reweighting:
it only chooses WHERE in phase space the training points sit. CPU only.

Momentum construction (massless 2->3, CM frame):
  E_i = x_i √s / 2;  three momenta are coplanar with Σp=0, opening angle from
  cos θ_ij = (E_k² - E_i² - E_j²)/(2 E_i E_j); a uniform SO(3) rotation orients the
  qqbar-g plane isotropically w.r.t. the fixed beams (the flat-solid-angle measure
  that multiplies the Dalitz density to give the full phase space).
"""
import argparse
import os
import sys

import numpy as np

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import mg5_pipeline_final as mp  # noqa
sys.path.insert(0, os.path.join(REPO, "analysis/divergences"))
from extract_ir import ir_observables  # noqa

WORK = os.environ["WORK"]
STANDALONE = f"{WORK}/mg5amcnlo/ee_uug_standalone"
PDG = np.array([11, -11, 2, -2, 21], dtype=int)   # e- e+ u ubar g  (row order)
# Lowered fiducial cuts (production uses pt=10,m=10,dr=0.4): open the sub-1e-6 IR.
LOW_CUTS = {"pt_min": 1.0, "cos_max": 0.9, "dr_min": 0.05, "m_min": 0.3}


def _random_so3(n, rng):
    """n uniform rotation matrices (3,3) via quaternions (Shoemake)."""
    u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
    q0 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q1 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q2 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q3 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (q2**2 + q3**2); R[:, 0, 1] = 2 * (q1*q2 - q0*q3); R[:, 0, 2] = 2 * (q1*q3 + q0*q2)
    R[:, 1, 0] = 2 * (q1*q2 + q0*q3); R[:, 1, 1] = 1 - 2 * (q1**2 + q3**2); R[:, 1, 2] = 2 * (q2*q3 - q0*q1)
    R[:, 2, 0] = 2 * (q1*q3 - q0*q2); R[:, 2, 1] = 2 * (q2*q3 + q0*q1); R[:, 2, 2] = 1 - 2 * (q1**2 + q2**2)
    return R


def _antenna_draw(nb, sqrts_min, sqrts_max, y_lo, rng):
    """One antenna proposal batch -> (P (m,5,4), sqrts (m,)). a,b log-uniform in
    [y_lo,1]; keep a+b<1 (physical Dalitz, x_g=a+b<1). Momenta built in CM frame."""
    sqrts = rng.uniform(sqrts_min, sqrts_max, nb)
    log_lo = np.log(y_lo)
    a = np.exp(rng.uniform(log_lo, 0.0, nb))      # 1-x_u   = y(ubar,g)
    b = np.exp(rng.uniform(log_lo, 0.0, nb))      # 1-x_ubar= y(u,g)
    ok = (a + b) < 1.0
    a, b, sqrts = a[ok], b[ok], sqrts[ok]
    m = len(a)
    x_u, x_ub, x_g = 1 - a, 1 - b, a + b          # energy fractions (sum=2)
    E = np.stack([x_u, x_ub, x_g], axis=1) * (sqrts[:, None] / 2.0)   # (m,3) E3,E4,E5
    E3, E4, E5 = E[:, 0], E[:, 1], E[:, 2]
    cos34 = np.clip((E5**2 - E3**2 - E4**2) / (2 * E3 * E4), -1.0, 1.0)
    sin34 = np.sqrt(1 - cos34**2)
    p = np.zeros((m, 3, 3))                        # 3-momenta of u, ubar, g
    p[:, 0, 0] = E3                                # u along +x
    p[:, 1, 0] = E4 * cos34; p[:, 1, 1] = E4 * sin34   # ubar in x-y plane
    p[:, 2] = -(p[:, 0] + p[:, 1])                # g closes the triangle
    R = _random_so3(m, rng)
    p = np.einsum("mij,mkj->mki", R, p)           # isotropic orientation
    finals = np.concatenate([E[:, :, None], p], axis=2)   # (m,3,4) E,px,py,pz
    Eb = sqrts / 2.0; z = np.zeros(m)
    beams = np.stack([np.stack([Eb, z, z, Eb], 1), np.stack([Eb, z, z, -Eb], 1)], axis=1)
    P = np.concatenate([beams, finals], axis=1)   # (m,5,4)
    return P, sqrts


def sample_antenna(n_events, sqrts_min, sqrts_max, y_lo, cuts, rng):
    """Antenna importance sampling with fiducial cuts, oversample-and-reject."""
    def draw(nb):
        P, sq = _antenna_draw(int(nb * 1.4) + 64, sqrts_min, sqrts_max, y_lo, rng)
        return P, sq
    P, sq = mp._collect_with_cuts(draw, n_events, [0.0, 0.0, 0.0], PDG, cuts)
    return [(P[i], PDG) for i in range(n_events)], sq


def sample_uniform(n_events, sqrts_min, sqrts_max, cuts, rng):
    """Flat RAMBO baseline (mg5 pipeline), same cuts."""
    return mp.sample_nbody_phase_space(
        n_events, sqrts_min, sqrts_max, [0.0, 0.0, 0.0], PDG, rng=rng, cuts=cuts)


def sample_mixture(n_events, sqrts_min, sqrts_max, y_lo, cuts, rng, frac_antenna=0.5):
    """frac_antenna of the events from the antenna sampler (fills the singular decades),
    the rest flat RAMBO (keeps the O(1) bulk). One dataset, shuffled together — so the
    model sees both the pole AND the bulk at fixed event count."""
    n_ant = int(round(frac_antenna * n_events))
    n_uni = n_events - n_ant
    ev_a, sq_a = sample_antenna(n_ant, sqrts_min, sqrts_max, y_lo, cuts, rng)
    ev_u, sq_u = sample_uniform(n_uni, sqrts_min, sqrts_max, cuts, rng)
    events = ev_a + ev_u
    sqrts = np.concatenate([sq_a, sq_u])
    idx = rng.permutation(n_events)
    return [events[i] for i in idx], sqrts[idx]


def build(events, sqrts):
    """Exact tree |M|^2 via the compiled C++ standalone (fixed α_s=0.118)."""
    with mp.CppDriverPipe(f"{STANDALONE}/driver", STANDALONE) as pipe:
        me2 = pipe.compute(events)
    mom = np.array([e[0].flatten() for e in events], dtype=np.float64)   # (N,20)
    pdg_block = np.tile(PDG.astype(np.float64), (len(events), 1))         # (N,5)
    rows = np.concatenate([mom, pdg_block, me2.reshape(-1, 1)], axis=1)   # (N,26)
    return rows, me2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["uniform", "antenna", "mixture"])
    ap.add_argument("--n", type=int, default=400000)
    ap.add_argument("--sqrts_min", type=float, default=91.0)
    ap.add_argument("--sqrts_max", type=float, default=1000.0)
    ap.add_argument("--y_lo", type=float, default=1e-7, help="antenna log-uniform floor")
    ap.add_argument("--mix_frac", type=float, default=0.5, help="mixture: antenna fraction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="output .npy (dataset) or .npz (test)")
    ap.add_argument("--as_test", action="store_true",
                    help="write an npz test set (rows+y_min+x_gmin+cut+pdg) for eval_heldout")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"mode={args.mode} n={args.n} sqrts=[{args.sqrts_min},{args.sqrts_max}] "
          f"cuts={LOW_CUTS} seed={args.seed}", flush=True)
    if args.mode == "antenna":
        events, sqrts = sample_antenna(args.n, args.sqrts_min, args.sqrts_max,
                                       args.y_lo, LOW_CUTS, rng)
    elif args.mode == "mixture":
        print(f"  mixture: antenna_frac={args.mix_frac}", flush=True)
        events, sqrts = sample_mixture(args.n, args.sqrts_min, args.sqrts_max,
                                       args.y_lo, LOW_CUTS, rng, args.mix_frac)
    else:
        events, sqrts = sample_uniform(args.n, args.sqrts_min, args.sqrts_max,
                                       LOW_CUTS, rng)
    rows, me2 = build(events, sqrts)

    P = 5
    mom = rows[:, :P * 4].reshape(-1, P, 4)
    obs = ir_observables(mom, PDG)
    y = obs["y_min"]
    for lo, hi in [(0, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3),
                   (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.0)]:
        n = int(((y >= lo) & (y < hi)).sum())
        print(f"  y_min[{lo:.0e},{hi:.0e}) : {n:7d}", flush=True)
    print(f"  |M|^2 in [{me2.min():.3e},{me2.max():.3e}]  log10 range={np.log10(me2.max()/me2.min()):.1f}",
          flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.as_test:
        np.savez_compressed(args.out, rows=rows.astype(np.float32),
                            y_min=y, x_gmin=obs["x_gmin"],
                            cut=np.array(1.0), region=np.array(args.mode), pdg=PDG)
    else:
        np.save(args.out, rows.astype(np.float32))
    print(f"saved -> {args.out}  (N={len(rows)})", flush=True)


if __name__ == "__main__":
    main()
