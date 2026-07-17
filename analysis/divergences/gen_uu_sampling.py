#!/usr/bin/env python
"""Deeper/cleverer phase-space GENERATION for the ee->uu s-channel Z resonance (L0).

Baseline generation is flat RAMBO: sqrt(s) ~ U[91,1000] (sample_2to2_phase_space), so the
Z pole at sqrt(s)~=91 (|M|^2 peaks ~200x, width Gamma_Z~2.5 GeV) is starved -- only ~1% of
events below 100 GeV, ~0.3% in [88,94]. Resampling the fixed RAMBO pool cannot fix this
(coverage ceiling: only ~3300 real pole events; resampling just duplicates them). We must
GENERATE fresh pole events, exactly as gen_uug_sampling.py generates fresh deep-IR points.

The general, process-agnostic target (Thread D) is flat in log|M|^2 -- equal training density
per amplitude decade. We cannot draw a target |M|^2, so:
  (1) generate a large CANDIDATE pool from a pole-covering proposal in sqrt(s) (a log-uniform
      offset component 91+exp(U) places dense candidates right at threshold, the rest uniform);
  (2) LABEL every candidate with the exact tree |M|^2 (ee_uu compiled C++ standalone, same
      convention as production -- verified matching |M|^2 range/scale);
  (3) THIN to flat-in-log|M|^2 on the TRUE labels (histogram inverse-density, WITHOUT
      replacement -> every kept event is a fresh, UNIQUE generated point).
Only |M|^2 shapes the density; m_Z / sqrt(s) are never used by the shaper (sqrt(s) is used
only at eval, to prove the fix is structure-agnostic). Since we fit log|M|^2 pointwise,
importance sampling needs NO target reweighting -- only WHERE the training points sit.

modes:
  uniform  : sqrt(s) ~ U[91,1000]                 -> RAMBO baseline (regenerates production)
  flatlogm : candidate pool -> label -> thin      -> flat per amplitude decade (pole covered)
"""
import argparse
import os
import sys

import numpy as np

REPO = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
sys.path.insert(0, REPO)
import mg5_pipeline_final as mp  # noqa

STANDALONE = f"{os.environ['WORK']}/mg5amcnlo/ee_uu_standalone"
PDG = np.array([11, -11, 2, -2], dtype=int)     # e- e+ u ubar (row order matches sample_2to2)


def build_momenta(sqrts, rng):
    """2->2 ee->uu massless kinematics for a given sqrt(s) array. Beams back-to-back along z;
    u/ubar back-to-back at (cos theta, phi) uniform on the sphere. Returns (N,4,4)."""
    n = len(sqrts)
    cos_t = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    Eb = sqrts / 2.0
    p_mag = Eb                                   # massless final state: |p| = E = sqrt(s)/2
    sin_t = np.sqrt(1.0 - cos_t ** 2)
    px = p_mag * sin_t * np.cos(phi)
    py = p_mag * sin_t * np.sin(phi)
    pz = p_mag * cos_t
    z = np.zeros(n)
    P = np.stack([
        np.stack([Eb, z, z,  Eb], axis=1),       # beam-
        np.stack([Eb, z, z, -Eb], axis=1),       # beam+
        np.stack([Eb, px, py, pz], axis=1),       # u
        np.stack([Eb, -px, -py, -pz], axis=1),    # ubar
    ], axis=1)                                    # (n,4,4)
    return P


def label(P):
    """Exact tree |M|^2 via the compiled ee_uu standalone."""
    events = [(P[i], PDG) for i in range(len(P))]
    with mp.CppDriverPipe(f"{STANDALONE}/driver", STANDALONE) as pipe:
        me2 = np.asarray(pipe.compute(events), dtype=np.float64)
    return me2


def candidate_sqrts(n, smin, smax, frac_pole, floor, rng):
    """Pole-covering proposal: frac_pole log-uniform in the threshold offset (dense at smin,
    where the resonance lives), the rest uniform (fills the bulk). Structure-light: it only
    needs to PLACE candidates everywhere incl. the pole; the label-based thinning does the
    flat-log|M|^2 shaping. floor = smallest offset above smin (GeV)."""
    n_pole = int(round(frac_pole * n))
    off = np.exp(rng.uniform(np.log(floor), np.log(smax - smin), n_pole))
    s_pole = smin + off
    s_uni = rng.uniform(smin, smax, n - n_pole)
    s = np.concatenate([s_pole, s_uni])
    return s[rng.permutation(len(s))]


def flat_logm_thin(me2, n, bins, rng):
    """Indices selecting n events flat-in-log|M|^2 (equal mass per equal-width log bin),
    WITHOUT replacement -> unique fresh events. Bins with too few candidates cap out (honest
    coverage report)."""
    u = np.log(me2)
    counts, edges = np.histogram(u, bins=bins)
    which = np.clip(np.digitize(u, edges[1:-1]), 0, len(counts) - 1)
    w = np.where(counts[which] > 0, 1.0 / counts[which].astype(float), 0.0)
    pi = w / w.sum()
    idx = rng.choice(len(me2), size=n, replace=False, p=pi)
    return idx


def _decade_report(me2, sqrts, tag):
    l10 = np.log10(me2)
    lo, hi = np.floor(l10.min()), np.ceil(l10.max())
    for e in np.arange(lo, hi):
        m = (l10 >= e) & (l10 < e + 1)
        print(f"    log10|M|^2 [{int(e):+d},{int(e)+1:+d}): {100*m.mean():5.1f}%  "
              f"(sqrt(s) med {np.median(sqrts[m]) if m.any() else float('nan'):.0f})", flush=True)
    print(f"    sqrt(s) pole frac [88,95)={100*np.mean((sqrts>=88)&(sqrts<95)):.2f}%  "
          f"<100={100*np.mean(sqrts<100):.2f}%", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["uniform", "flatlogm"])
    ap.add_argument("--n", type=int, default=400000, help="events in the output training set")
    ap.add_argument("--n_cand", type=int, default=3000000, help="flatlogm: candidate pool size")
    ap.add_argument("--sqrts_min", type=float, default=91.0)
    ap.add_argument("--sqrts_max", type=float, default=1000.0)
    ap.add_argument("--frac_pole", type=float, default=0.5, help="candidate frac from the pole proposal")
    ap.add_argument("--floor", type=float, default=0.02, help="min sqrt(s) offset above threshold (GeV)")
    ap.add_argument("--bins", type=int, default=40, help="log|M|^2 bins for flat thinning")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="output .npy")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    print(f"mode={args.mode} n={args.n} sqrts=[{args.sqrts_min},{args.sqrts_max}] seed={args.seed}", flush=True)

    if args.mode == "uniform":
        sqrts = rng.uniform(args.sqrts_min, args.sqrts_max, args.n)
        P = build_momenta(sqrts, rng)
        me2 = label(P)
    else:
        print(f"  generating {args.n_cand} candidates (frac_pole={args.frac_pole}, floor={args.floor})", flush=True)
        s_cand = candidate_sqrts(args.n_cand, args.sqrts_min, args.sqrts_max, args.frac_pole, args.floor, rng)
        P_cand = build_momenta(s_cand, rng)
        me2_cand = label(P_cand)
        print(f"  labeled {len(me2_cand)} candidates; |M|^2 log10 span "
              f"{np.log10(me2_cand.min()):.2f}..{np.log10(me2_cand.max()):.2f}", flush=True)
        idx = flat_logm_thin(me2_cand, args.n, args.bins, rng)
        P, me2, sqrts = P_cand[idx], me2_cand[idx], s_cand[idx]

    _decade_report(me2, sqrts, args.mode)
    mom = P.reshape(len(P), -1)                                   # (N,16)
    pdg_block = np.tile(PDG.astype(np.float64), (len(P), 1))       # (N,4)
    rows = np.concatenate([mom, pdg_block, me2.reshape(-1, 1)], axis=1)   # (N,21)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, rows.astype(np.float64))
    print(f"saved -> {args.out}  (N={len(rows)}, shape {rows.shape})", flush=True)


if __name__ == "__main__":
    main()
