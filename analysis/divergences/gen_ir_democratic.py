#!/usr/bin/env python
"""Process-AGNOSTIC IR-democratic phase-space generation.

The base proposal for sigma-driven online generation must reach the singular regions of ANY
massless process WITHOUT a process-specific antenna/channel list -- otherwise the "coordinate-free"
method has smuggled the physics back in. The only generic fact used here is that IR singularities of
a massless amplitude live at SMALL INVARIANTS: soft (some energy E_i -> 0) and collinear (some pair
invariant s_ij=(p_i+p_j)^2 -> 0). Both are reached democratically by a recursive 1->2 splitting of
the total invariant mass with the intermediate masses drawn FLAT IN LOG (the process-agnostic core of
the SARGE/HAAG antenna generators).

Construction (CM frame, N final-state particles of mass m_i, total energy sqrt(s)):
  blob starts as Q=(sqrt(s),0,0,0), invariant mass M_N=sqrt(s). For k=N..2 split the blob (mass M_k)
  into particle p_k (mass m_k) + a remaining blob (mass M_{k-1}): draw M_{k-1}^2 LOG-UNIFORM in
  [(m_rest+Mfloor)^2, (M_k-m_k)^2] where m_rest=sum of the still-unemitted masses, so a small M_{k-1}
  makes the remaining cluster soft/collinear. Two-body kinematics give back-to-back momenta in the
  blob rest frame at an isotropic direction; boost to the lab by the blob's lab momentum. The last
  blob IS p_1. Momentum conservation and on-shell-ness are EXACT by construction. Mfloor=sqrt(s)*
  sqrt(y_lo) sets the deepest invariant reached (analog of the antenna's y_lo).

We DO NOT need this proposal's density/Jacobian: training regresses log|M|^2 pointwise, so only WHERE
the events sit matters (same argument as gen_uug_sampling.py). So any construction that yields valid
on-shell momentum-conserving events covering the corners is admissible -- this one, unlike a bespoke
antenna, is process-agnostic.

CPU only. Labeling (exact tree |M|^2) is a separate step via the compiled standalone.
"""
import argparse
import os
import sys

import numpy as np

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)


# The splitter now lives in the pipeline (mg5_pipeline_final: sampling policy); one copy.
from mg5_pipeline_final import _random_dirs, _boost_from_rest, _two_body, democratic_draw  # noqa: E402,F401


def build_full_event(P_final, sqrts, beam_pdg=(11, -11)):
    """Prepend the two back-to-back beams to the final-state momenta -> (nb, 2+N, 4)."""
    nb = len(sqrts)
    Eb = sqrts / 2.0; z = np.zeros(nb)
    beams = np.stack([np.stack([Eb, z, z, Eb], 1), np.stack([Eb, z, z, -Eb], 1)], axis=1)
    return np.concatenate([beams, P_final], axis=1)


# ---------------------------------------------------------------- generic IR diagnostics (any N)
def ir_report(P, pdg, tag=""):
    """Print soft (min energy fraction) + collinear (min pair invariant) coverage per decade,
    process-agnostically over all colored (massless-parton) final legs."""
    P = np.asarray(P)
    apdg = np.abs(np.asarray(pdg))
    Q = P[:, 0] + P[:, 1]
    def dot(a, b): return a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)
    s = dot(Q, Q)
    fin = np.where((apdg <= 6) | (apdg == 21))[0]                 # colored/massless final legs
    fin = fin[fin >= 2]
    # soft: energy fraction x_i = 2 E_i / sqrt(s) in CM (Q ~ (sqrt(s),0,0,0) here)
    xs = np.stack([2.0 * P[:, i, 0] / np.sqrt(s) for i in fin], 1)
    x_min = xs.min(1)
    # collinear: y_ij = s_ij / s over final pairs
    pairs = [(fin[a], fin[b]) for a in range(len(fin)) for b in range(a + 1, len(fin))]
    if pairs:
        yv = np.stack([dot(P[:, i] + P[:, j], P[:, i] + P[:, j]) / s for i, j in pairs], 1)
        y_min = np.clip(yv, 1e-14, None).min(1)
    else:
        y_min = np.ones(len(P))
    print(f"  [{tag}] N_final={len(fin)}  pairs={pairs}", flush=True)
    for name, v in [("y_min (collinear)", y_min), ("x_min (soft)", x_min)]:
        line = "    ".join(
            f"{lo:.0e}-{hi:.0e}:{100*np.mean((v>=lo)&(v<hi)):4.1f}%"
            for lo, hi in [(0,1e-6),(1e-6,1e-5),(1e-5,1e-4),(1e-4,1e-3),(1e-3,1e-2),(1e-2,1e-1),(1e-1,1.01)])
        print(f"    {name:20s} {line}", flush=True)
    # momentum-conservation + on-shell sanity
    dP = (P[:, 2:].sum(1) - Q)
    mom_err = np.abs(dP).max()
    m2 = dot(P[:, fin], P[:, fin])
    onshell = np.abs(m2).max()
    print(f"    momentum-conservation max|dP|={mom_err:.2e}   on-shell max|p^2|={onshell:.2e}", flush=True)


PDG_TABLE = {
    "ee_uu":   [11, -11, 2, -2],
    "ee_uug":  [11, -11, 2, -2, 21],
    "ee_uugg": [11, -11, 2, -2, 21, 21],
}
MASS = {11: 0.0, -11: 0.0, 2: 0.0, -2: 0.0, 21: 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="ee_uugg", choices=list(PDG_TABLE))
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--sqrts_min", type=float, default=91.0)
    ap.add_argument("--sqrts_max", type=float, default=1000.0)
    ap.add_argument("--y_lo", type=float, default=1e-7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validate", action="store_true", help="run the same code on uu/uug/uugg and report IR coverage")
    args = ap.parse_args()

    if args.validate:
        for proc in ("ee_uu", "ee_uug", "ee_uugg"):
            pdg = np.array(PDG_TABLE[proc], int)
            masses = np.array([MASS[p] for p in PDG_TABLE[proc][2:]], float)
            rng = np.random.default_rng(args.seed)
            sqrts = rng.uniform(args.sqrts_min, args.sqrts_max, args.n)
            Pf = democratic_draw(args.n, sqrts, masses, args.y_lo, rng)
            P = build_full_event(Pf, sqrts)
            print(f"\n=== {proc} (same democratic_draw, N_final={len(masses)}) ===", flush=True)
            ir_report(P, pdg, tag=proc)
        return

    pdg = np.array(PDG_TABLE[args.proc], int)
    masses = np.array([MASS[p] for p in PDG_TABLE[args.proc][2:]], float)
    rng = np.random.default_rng(args.seed)
    sqrts = rng.uniform(args.sqrts_min, args.sqrts_max, args.n)
    Pf = democratic_draw(args.n, sqrts, masses, args.y_lo, rng)
    P = build_full_event(Pf, sqrts)
    ir_report(P, pdg, tag=args.proc)


if __name__ == "__main__":
    main()
