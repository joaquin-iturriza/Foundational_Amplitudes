#!/usr/bin/env python3
"""
Parse eett.dat and eeuu.dat (NLO amplitude output) into the .npy format
used by this project.

Output .npy layout (one row per event):
    [E0 px0 py0 pz0 | E1 px1 py1 pz1 | E2 px2 py2 pz2 | E3 px3 py3 pz3 | pdg0 pdg1 pdg2 pdg3 | weight]

Convention (confirmed by email + analytic check):
    virt_fin is stored WITHOUT the α_s / (2π) prefactor and WITHOUT the EW coupling.
    The .dat files use the e=1 convention; MadGraph uses physical couplings.
    For eeuu: virt_fin / born = C_F(π²-8) = 2.4928... exactly, independent of
    kinematics — this is a pure color/kinematic coefficient, not a physical correction.

    EW coupling prefactor:
        e⁴ = (4π α_em)²,  α_em = 1 / AEW_INV = 1 / 132.507
        E4 = (4π / AEW_INV)²  ≈ 8.99e-3

    The correct NLO squared amplitude (matching MadGraph LO normalization) is:
        weight = (born + (α_s(√s) / (2π)) × virt_fin) × e⁴

    where α_s is evaluated at μ_R = √s per event (matching the .dat file convention,
    muR² = s).

Two output files are produced per process:
    *_nlo_amplitudes.npy   — full NLO weight = (born + (αs/2π)·virt) × e⁴
    *_nlo_virt_e4.npy      — virtual-only    = virt_fin × e⁴  (no αs factor)
                             Add (αs/2π) × virt_e4 to born×e⁴ to reconstruct NLO.

Particle ordering in .dat files:
    p1 = e-  (PDG  11)  — massless, +z beam
    p2 = e+  (PDG -11)  — massless, -z beam
    p3, p4   — outgoing quark pair

IMPORTANT (ttbar t<->tbar labeling, established 2026-06 by point-by-point
comparison vs MadGraph):
    For eett the outgoing pair is ordered  p3 = tbar (-6),  p4 = t (6)  —
    the OPPOSITE of the naive p3=t, p4=tbar guess. Evidence: MadGraph's Born for
    "e+ e- > t t~" reproduces the .dat born only when p3 is assigned the antitop;
    with p3=t the Born ratio scatters by ~110% (anti-correlated with cos(theta_t)),
    with p3=tbar it is flat to 0.25%. For massless eeuu the Born is ~forward-
    backward symmetric so the analogous choice is immaterial (and we keep p3=u).
    The amplitude is C-invariant, so this is purely a labeling convention; we pick
    e- along +z (standard) which forces p3=tbar, p4=t for ttbar.
"""

import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent

# EW coupling: aEWM1 = 1/α_em from param_card.dat (same value used by MadGraph)
AEW_INV = 132.507
E4 = (4.0 * np.pi / AEW_INV) ** 2   # e⁴ = (4π α_em)²  ≈ 8.99e-3

PROCESSES = {
    "eett": {
        "input":       DATA_DIR / "eett.dat",
        "output_nlo":  DATA_DIR / "ee_ttbar_nlo_amplitudes.npy",
        "output_virt": DATA_DIR / "ee_ttbar_nlo_virt_e4.npy",
        # p3 = tbar (-6), p4 = t (6): see header note (confirmed vs MadGraph Born).
        "pdg_ids":     [11, -11, -6, 6],
    },
    "eeuu": {
        "input":       DATA_DIR / "eeuu.dat",
        "output_nlo":  DATA_DIR / "ee_uu_nlo_amplitudes.npy",
        "output_virt": DATA_DIR / "ee_uu_nlo_virt_e4.npy",
        "pdg_ids":     [11, -11, 2, -2],
    },
}

# .dat columns: id muR2 p1E p1px p1py p1pz p2E p2px p2py p2pz
#               p3E p3px p3py p3pz p4E p4px p4py p4pz born virt_fin
# indices:       0   1   2   3   4   5   6   7   8   9
#               10  11  12  13  14  15  16  17  18  19
MOM_COLS  = list(range(2, 18))
BORN_COL  = 18
VIRT_COL  = 19
P1E_COL   = 2    # √s = 2 * p1E


def alphas_1loop(sqrts, alphas_mz=0.118, mz=91.1876, nf=5):
    """α_s(μ) at 1-loop, μ = sqrts."""
    b0 = (11 * 3 - 2 * nf) / (12 * np.pi)
    return alphas_mz / (1.0 + alphas_mz * b0 * 2.0 * np.log(sqrts / mz))


def make_arr(raw, weight, pdg_ids):
    n = raw.shape[0]
    arr = np.empty((n, 21), dtype=np.float64)
    arr[:, :16] = raw[:, MOM_COLS]
    arr[:, 16] = pdg_ids[0]
    arr[:, 17] = pdg_ids[1]
    arr[:, 18] = pdg_ids[2]
    arr[:, 19] = pdg_ids[3]
    arr[:, 20] = weight
    return arr


print(f"e⁴ = {E4:.6e}  (1/e⁴ = {1/E4:.3f})")

for proc_name, proc in PROCESSES.items():
    print(f"\nProcessing {proc_name} ...")

    raw    = np.loadtxt(proc["input"], comments="#", dtype=np.float64)
    born   = raw[:, BORN_COL]
    virt   = raw[:, VIRT_COL]
    sqrts  = 2.0 * raw[:, P1E_COL]
    alphas = alphas_1loop(sqrts)

    # Sanity-check for eeuu: ratio should equal C_F(π²-8) = 2.4928...
    if proc_name == "eeuu":
        cf_pi2_8 = (4/3) * (np.pi**2 - 8)
        ratio    = (virt / born).mean()
        print(f"  virt/born = {ratio:.6f}  (expected C_F(π²-8) = {cf_pi2_8:.6f})")
        correction_pct = ((alphas / (2*np.pi)) * virt / born * 100).mean()
        print(f"  mean NLO correction = {correction_pct:.2f}% of Born")

    # ── full NLO amplitude ───────────────────────────────────────────────
    weight_nlo = (born + (alphas / (2.0 * np.pi)) * virt) * E4
    arr_nlo = make_arr(raw, weight_nlo, proc["pdg_ids"])
    np.save(proc["output_nlo"], arr_nlo)
    w = arr_nlo[:, 20]
    print(f"  NLO → {proc['output_nlo'].name}  shape={arr_nlo.shape}")
    print(f"    weight range: [{w.min():.4e}, {w.max():.4e}]  "
          f"negative: {(w < 0).sum()} ({100*(w<0).mean():.2f}%)")

    # ── virtual-only × e⁴  (no αs factor, for flexible combination) ─────
    weight_virt = virt * E4
    arr_virt = make_arr(raw, weight_virt, proc["pdg_ids"])
    np.save(proc["output_virt"], arr_virt)
    wv = arr_virt[:, 20]
    print(f"  Virt → {proc['output_virt'].name}  shape={arr_virt.shape}")
    print(f"    virt_e4 range: [{wv.min():.4e}, {wv.max():.4e}]")
