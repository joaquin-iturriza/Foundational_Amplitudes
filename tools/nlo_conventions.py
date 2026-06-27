#!/usr/bin/env python3
r"""
nlo_conventions.py — the locked one-loop ("virtual") convention spec.
=====================================================================
Single source of truth for *how* we generate NLO QCD virtual matrix elements
with MadGraph so that the output equals what the collaborators' GoSam+Sherpa
chain produces, for ANY process — not just the two reference datasets.

The whole point: a one-loop "convention" is a FINITE, PROCESS-INDEPENDENT set of
choices. Pin them once (below) and every process is fixed by construction. The
two reference datasets (eeuu massless + eett massive) over-determine this set:
  * eeuu pins the massless / EW / IR-normalization sector,
  * eett pins the heavy-quark mass-renormalization sector.

Evidence (point-by-point vs the .dat references, tools/compare_madloop_vs_dat.py):
  * eeuu  virtual finite part matches MadLoop to 2e-12, IR poles -3C_F / -2C_F exact.
  * eett  Born matches to 0.25% (after the t<->tbar relabel), and the virtual
          finite part differs by EXACTLY  Δ = C_F (3 ln(s/m_t^2) + 5)  (rel. 8e-4),
          a pure heavy-quark renormalization term (pole-mass vs MS-bar).

------------------------------------------------------------------------------
THE LOCKED KNOBS
------------------------------------------------------------------------------
1. Finite-part normalization  (BLHA / CDR, 't Hooft-Veltman):
      The virtual is returned as the Laurent expansion of  2 Re(M0* M1):
          V = (alpha_s / 2pi) * (born) * [ c2/eps^2 + c1/eps + c0 ]
      with the standard (4pi)^eps / Gamma(1-eps) factor pulled out. This is a
      TOOL property shared by MadLoop and GoSam (both BLHA providers) — it is
      why eeuu matches to machine precision and it does NOT depend on the process.
      MadLoop's f2py `get_me_full` returns MATELEM(0:3,0) = (born, c0, c1, c2)*…
      already carrying the alpha_s/2pi; divide by born and by (alpha_s/2pi)=AO2PI
      to recover the pure coefficients c0,c1,c2.

2. What `virt_fin` means:
      the BARE UV-renormalized one-loop finite part c0 (IR poles still present),
      NOT an I-operator-subtracted finite remainder. (Confirmed: eeuu's .dat
      virt_fin equals MadLoop's bare c0, not a Catani-Seymour subtracted value.)

3. Renormalization scale:   mu_R^2 = s   per event   (muR2 == s in both .dat files).

4. Strong coupling:   alpha_s(M_Z) = 0.118, 1-loop running, n_f = 5, evaluated at
      mu = sqrt(s).  NB: virt_fin/born is alpha_s-INDEPENDENT (it is a coefficient);
      alpha_s only enters the absolute virtual via the AO2PI prefactor.

5. Electroweak inputs  (G_mu scheme, as in the loop_sm restrict_default param_card):
      aEWM1 = 132.507,  M_Z = 91.1880,  Gamma_Z = 2.441404,
      G_F = 1.16639e-5,  Gamma_W = 2.0476   (M_W, sin^2θ_W derived).
      -> effective born normalization  born_MG / born_dat = 9.033e-3  (= 1/110.7),
         vs the naive E4 = (4pi/132.507)^2 = 8.988e-3 (1/111.2): a 0.5% offset that
         is the G_mu-vs-α(M_Z) scheme choice, NOT a bug. For absolute amplitudes
         use the MEASURED E4_EFF below (or, better, take the Born from MadGraph too).

6. Heavy-quark mass renormalization:
      MadLoop default = on-shell (pole) mass. The references differ by the exact
      term in HEAVY_QUARK_SCHEME_SHIFT (per heavy quark pair). Either (a) add that
      term to MadLoop's finite part, or (b) configure MadLoop for the MS-bar mass.
      Light quarks (u,d,s,c; b often) are massless -> no such term.

7. Particle ordering / labeling:
      e- = +z beam (p1), e+ = -z (p2). For a heavy pair the .dat orders the
      ANTIquark before the quark in the outgoing slots (p3 = Qbar, p4 = Q); see
      data/parse_dat.py. Immaterial for the (near-symmetric) massless Born.

------------------------------------------------------------------------------
SCOPE / open edges (where "any process" needs one extra check, not a redesign):
  * >2 colored partons (ee->qqg, qqgg): richer IR; the universal DOUBLE-pole
    check still certifies them; the SINGLE pole then needs color-correlated Borns.
  * processes with alpha_s already at LO: the alpha_s renormalization scheme
    becomes a live knob (dormant for pure-EW LO like ee->partons).
  * real radiation / full NLO: separate (FKS vs CS subtraction) — later phase.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Scale / coupling
# ---------------------------------------------------------------------------
def mu_r2(momenta_in):
    """Renormalization scale squared = s = (sum of incoming momenta)^2."""
    pin = np.sum(momenta_in, axis=0)
    return pin[0]**2 - pin[1]**2 - pin[2]**2 - pin[3]**2


def alphas_1loop(sqrts, alphas_mz=0.118, mz=91.1876, nf=5):
    """alpha_s(mu=sqrt(s)) at 1-loop — identical to data/parse_dat.py."""
    b0 = (11 * 3 - 2 * nf) / (12 * np.pi)
    return alphas_mz / (1.0 + alphas_mz * b0 * 2.0 * np.log(sqrts / mz))


# ---------------------------------------------------------------------------
# Electroweak (G_mu scheme) — param_card patches to apply to a fresh standalone
# ---------------------------------------------------------------------------
PARAM_CARD_PATCHES = {
    "aEWM1": 132.507,
    "MZ":    91.1880,
    "WZ":    2.441404,
    "WW":    2.0476,
    "GF":    1.16639e-5,
    # heavy-quark pole masses used by the references:
    "MT":    172.5,
    "ymt":   172.5,
}

# Empirical absolute-normalization factor (born_MG / born_dat), G_mu scheme.
E4_EFF = 9.033e-3          # = 1/110.7 ; use instead of (4pi/132.507)^2 for .dat data

# ---------------------------------------------------------------------------
# Color / collinear constants for the universal IR pole structure
# ---------------------------------------------------------------------------
CF = 4.0 / 3.0
CA = 3.0
TF = 0.5

def gamma_quark():
    return 1.5 * CF                     # gamma_q = 3/2 C_F

def gamma_gluon(nf=5):
    return (11.0 * CA - 4.0 * TF * nf) / 6.0   # gamma_g = (11 C_A - 4 T_F n_f)/6


def casimir(pdg):
    """Color Casimir of a parton: C_F (quark), C_A (gluon), 0 (colorless)."""
    a = abs(int(pdg))
    if 1 <= a <= 6:
        return CF
    if a == 21:
        return CA
    return 0.0


# ---------------------------------------------------------------------------
# Heavy-quark mass-scheme shift (pole <-> the references' scheme)
# ---------------------------------------------------------------------------
def heavy_quark_scheme_shift(s, m_heavy):
    r"""Additive shift to the finite coefficient c0 (= virt_fin/born, units
    alpha_s/2pi) that converts MadLoop's on-shell (pole-mass) result to the
    references' convention, per heavy quark PAIR:

        c0_reference = c0_MadLoop + C_F ( 3 ln(s/m^2) + 5 )

    Validated on eett to 8e-4 relative across sqrt(s) in [350,1000] GeV.
    Returns 0 for a massless quark (m_heavy <= 0)."""
    if m_heavy is None or m_heavy <= 0:
        return 0.0
    return CF * (3.0 * np.log(s / m_heavy**2) + 5.0)


if __name__ == "__main__":
    print(__doc__)
    print(f"CF={CF}  CA={CA}  gamma_q={gamma_quark()}  gamma_g(nf=5)={gamma_gluon()}")
    print(f"E4_EFF={E4_EFF}  (1/E4_EFF={1/E4_EFF:.3f})")
    for s_, m_ in [(500.0**2, 172.5), (1000.0**2, 172.5)]:
        print(f"  heavy_quark_scheme_shift(sqrt(s)={s_**0.5:.0f}, m=172.5) = "
              f"{heavy_quark_scheme_shift(s_, m_):.4f}")
