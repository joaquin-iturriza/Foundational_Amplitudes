#!/usr/bin/env python
"""Synthetic matrix elements: the synthetic-amplitude prior (docs/results.tex, hand-off
"priors and representations").

A synthetic process ``syn_<k>`` (or ``syn<n>_<k>`` with n final particles forced) is a fixed
random structure drawn from a seed derived from its name, so the name alone regenerates it:

  * legs: a real SM initial pair (massless, along z) and n = 2..4 SM final particles,
    charge, baryon and lepton number conserved, at the generator masses of
    particle_ids.GLOBAL_PROPERTY_MATRIX, so the model's particle-property inputs are real;
  * K random tree diagrams (random cubic trees over the legs). Every internal edge is a
    propagator over a channel S (a leg subset; s_S = (sum of all-outgoing momenta)^2,
    s-type when S holds only final legs, t-type when it splits the two beams) carrying a
    real SM particle of the channel's charge: gamma/g/Z/H, W/e/mu, u/t, d/b. A massive
    s-type propagator has a Breit-Wigner width Gamma = r M, r log-uniform per particle; a
    t-type one has none. The scalar (PDG 25) takes a random mass per process half of the
    time (``internal_masses``), the others the generator masses. A massless or light (b)
    propagator enters as 1/D_S (scalar-like) or 1/sqrt(D_S) (the 1/s_ij of a gauge-theory
    collinear limit in |M|^2); a massless one only sits on channels the fiducial cuts keep
    away from s_S = 0;
  * per helicity h = 1..H and diagram d a complex coefficient c_hd and a random homogeneous
    polynomial P_hd in the pair invariants s_ij, times the power of s that gives the term
    the amplitude's mass dimension 4 - N:

        |M|^2 = g * < sum_h | sum_d c_hd P_hd(s_ij) s^(-k_hd/2) / prod_{S in d} D_S |^2 >_perm,
        D_S = s_S - M_S^2 + i M_S Gamma_S   (or its square root, |s_S| for a massless one),

    g = (4 pi alpha_s(sqrt s))^k_s (4 pi alpha)^(n - k_s) (alpha_s run per event at one loop,
    as the real pools), averaged over the permutations of
    identical legs (the target must be a function of the particle set). |M|^2 > 0 has mass
    dimension 8 - 2N and is Lorentz invariant by construction (only invariants enter).

That is version 1 (docs/results.tex, sec:synthetic). Versions 2-6 (site decisions D36, D38, D40; the "Proposed, not
run" paragraphs of that section) are cumulative, each keeping every change of the one before.

Names and versions. The name selects the version, so a recipe can mix them:

    syn_<k>, syn<n>_<k>          version 1 (SYN_VERSION, frozen: every v1 pool depends on its exact bytes)
    syn[<n>]v<V>_<k>             version V = 2..5, revision 1 (as first run; superseded, kept reproducible),
                                 n = 2..6 forced final particles or drawn when absent (v5: n = 5 or 6 only)
    syn[<n>]v<V>r2_<k>           version V = 2..5, revision 2 (below): e.g. syn5v3r2_00000, syn6v5r2_00002
    syn[<n>]v6_<k>               version 6 on version 4's general legs (n = 2..6, drawn 2..4 when absent)
    syn[<n>]v6t_<k>              version 6 on version 5's target shape (n = 5 or 6)

  The seed is (salt, V, n_forced, k) for revision 1 (V = 1 for the unsuffixed names) and (salt, V, n_forced, k,
  2, target-shaped) for revision 2 and version 6, so the name alone regenerates the process, different versions or
  revisions of one index are unrelated draws, and no revision-2 name can meet a revision-1 pool (the pools' cache
  key is the name). Version 1 keeps its own structure and evaluator code untouched; versions 2-5 are evaluated by
  evaluate_vec, version 6 by evaluate_v6.

  Revision 2 (2026-10-12, the user's correction: K drawn log-uniform up to the cap left typical processes with a
  handful of the hundreds of diagrams the versions were meant to have):
    v2r2  K = K_cap, not log-uniform up to it.
    v3r2+ every SM-legal tree of the process (version 3's vertices and light-line protection; legal_diagram_count
          gives q qbar -> V + 4g 384, + 5g 3840), listed by subset recursion (_legal_tables) and grouped into orbits
          of the identical-leg permutations; one numerator per orbit, and |M|^2 sums every labelled diagram of every
          orbit coherently (each orbit's |Pi| images weighted 1/|stabiliser|) instead of averaging |.|^2 over |Pi|.
          Lines a stabiliser maps onto each other share their drawn power. No cap on the diagram count (the user,
          2026-10-13: optimise before compromising); only above _ENUM_MAX = 50000 legal trees, which no recipe name
          reaches, do the orbits come from uniformly sampled trees (up to _ENUM_MAX labelled) instead of a listing.
  v6  (D40) Spinor-helicity numerators and the tree-level selection rules, on v5r2's diagrams (ggH vertex
      included; general legs as v4, or the target shape as v5 with the t tag). Every leg carries a helicity in the
      all-outgoing convention: fermions +-1/2, g and gamma +-1, Z and W -1, 0, +1, H 0. A massive leg's spinors are
      those of the light-cone component of its momentum along its direction in the centre-of-mass frame,
      p_flat = (E + |k|)/2 (1, k/|k|) there (p = p_flat + m^2/(2 p_flat.q) q, q null against it; p_flat -> p as
      m -> 0): the helicity quantised along the momentum in the CM frame, symmetric in the beams and the same in
      every term (a reference that changed between terms would mix helicity bases and break Lorentz invariance).
      The opposite component (1, -k/|k|) would put p_flat exactly on the other final particle of a 2->2 process. Per configuration and
      orbit representative a numerator c * a * monomial(<ij>, [ij]) * s^(-k/2) (_V6_MONOS = 1 monomial: the
      configuration sum already gives each diagram many independent random structures) with
        weights: angle minus square count of leg i = -2 h_i, so A goes to t^(-2 h_i) A when lambda_i goes to
                 t lambda_i and lambda~_i to lambda~_i / t;
        soft:    a gauge leg (massless final g, gamma) has as many brackets in the denominator as in the numerator,
                 so the monomial stays finite as it goes soft; its denominators pair with a beam (<k a>, never zero
                 inside the fiducial cuts), the eikonal shape <a i>[i k]/<a k>;
        collinear: a massless line on two massless final legs i, k keeps version 4's power 1 and its numerator has
                 exactly one <ik> or [ik] (a gauge leg's helicity fixes which), so |M|^2 ~ 1/s_ik and, with the
                 soft rule, ~ 1/E_k^2 (no W factor needed);
        dimension: the remaining stubs pair at random (no <ii>, no second bracket of a collinear pair), padded with
                 dipoles (one angle and one square stub on a non-gauge leg) to the bracket count n_b + 2 delta,
                 delta ~ U{0,1,2}; s^(-k/2) with k = n_b - 2E + N - 4 restores the mass dimension 4 - N.
      Selection rules: helicity is conserved along every massless fermion line (its two ends opposite in the
      all-outgoing convention), left-handed along a line through a W vertex or ending on a neutrino; with every
      external leg massless and no scalar, configurations with fewer than two negative or two positive helicities
      vanish. A diagram contributes only to the configurations its lines allow; a configuration no image allows is
      exactly zero. |M|^2 = (1/4) sum over configurations of (|A|^2 + |A^P|^2)/2, A^P the parity image (every
      angle bracket swapped with the square one, i.e. |A| at the spatially inverted momenta); a diagram allowed at
      both a configuration and its parity partner gives the partner its numerator conjugated (one draw per pair).
      Every legal diagram, as revision 2: no cap on the evaluated terms.

  v2  K grows with multiplicity. K = floor(exp U[0, ln(K_cap + 1))) (log-uniform on 1..K_cap),
          K_cap = max(K_max^v1(N), floor(_KFRAC (2N-5)!! / |Pi|)),   _KFRAC = 1/4,
      (2N-5)!! the number of cubic trees on N labelled legs, |Pi| the identical-leg permutations. The permutation
      average turns each drawn diagram into |Pi| labelled images, so K|Pi| <= (2N-5)!!/4 bounds the labelled
      diagrams evaluated, and the cost (linear in K|Pi|) with it; diagrams are drawn distinct up to the
      permutations. The fraction keeps the cap under the SM's own cubic-tree count of the target family
      (legal_diagram_count: q qbar -> V + 4g 384 = 0.41 (2N-5)!!, V + 5g 3840 = 0.37 (2N-5)!!) and puts at
      most ~2600 labelled terms per helicity in a 2->6 event; never below version 1's K_max. Couplings, legs,
      lines and numerators as version 1.
  v3  SM vertices and flavour. Legs also conserve quark generation (diagonal CKM) and lepton family. Trees are
      built bottom-up so that every vertex is an SM cubic vertex (sm_vertices: f fbar gamma for charged f,
      f fbar Z, q qbar g, f fbar H for b and t, the doublets' W vertices, W W gamma, W W Z, g g g, W W H, Z Z H);
      a line's particle is drawn among the legal ones (version 1's weights, _LINE_W), the light-line
      protection as before; a final state with no legal tree is redrawn. Couplings come from the vertices, per
      diagram: amplitude factor (4 pi alpha_s)^(n_QCD/2) (4 pi alpha)^(n_QED/2) with alpha_s run per event, so
      diagrams of different orders interfere as in a mixed-order process; the order vector holds the maxima
      over the diagrams (mg5_pipeline_final.order_vector's convention).
  v4  The collinear (and soft) limit by construction. A massless line on two massless final legs i, k keeps
      power 1 (version 1's 1/sqrt(s_ik) power is gone there), and every numerator term of its diagram carries
          B_ik = |s_ik|^(1/2) W_ik tr(i k b a) / |tr(i k b a)|    or its conjugate ([ik] vs <ik>, drawn per term),
          tr(i k b a) = <ik>[kb]<ba>[ai]   (beams a, b; weight 0 on every leg: Lorentz invariant),
          W_ik = prod over the gluons/photons x among i, k of ((q_i + q_k).P / q_x.P)^(1/2),  P = p_a + p_b,
      with s_ik left out of that term's monomials (brackets(), mass dimension counted in k). Then
      |M|^2 ~ 1/s_ik as i || k (the numerator vanishes like sqrt(s_ik), with the azimuthal phase of <ik>), and
      W makes a soft gluon or photon k give |M|^2 ~ 1/E_k^2 (the eikonal power; its coefficient is the random
      hard function, not Sum T_i.T_j p_i.p_j / (p_i.k p_j.k) times a lower-point synthetic amplitude, and
      initial-state (t-type) and multi-leg massless lines keep version 1's rule). |M|^2 is averaged with its
      parity image (every bracket conjugated), so it is parity even like an unpolarised QCD sum.
  v5  The target's shape: q qbar' -> V + (n-1) g, n = 5, 6 (2->5, 2->6); V = gamma, Z, W+, H by k % 4 (so any
      four consecutive indices hold all four), u ubar or d dbar at 1/2 each (u dbar for the W, as the catalog's
      udbar_W*); the effective g g H vertex (heavy-top limit, amplitude factor alpha_s e / (3 pi)), legal only on
      the external H, since the generator's u, d are massless; v4 rules otherwise.

The structure also gives the diagram sidecar (``<name>.diagrams.json``, tools/dump_diagrams.py
format), so data.offshell_per_event sees the true propagators. Generation goes through the
normal recipe path: mg5_pipeline_final registers ``syn*`` names into PROCESSES with
``kind: synthetic`` and labels events with ``label_events`` instead of a MadGraph backend.

CLI:
    python tools/synthetic_amplitudes.py --describe syn_00042 syn6v5r2_00002 syn6v6t_00001
    python tools/synthetic_amplitudes.py --recipe recipes/synthetic_pilot.yaml --first 0 --count 20 \
        --n-train 20000 --n-val 2000 --n-test 2000
    python tools/synthetic_amplitudes.py --sidecars recipes/synthetic_pilot.yaml [--out-dir data/diagrams]
"""
import functools
import hashlib
import itertools
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from particle_ids import PARTICLE_PROPERTIES, PARTICLE_FEATURE_NAMES, _MASSLESS  # noqa: E402

# The version of the UNSUFFIXED names (syn_<k>, syn<n>_<k>): frozen at 1, every v1 pool depends on it. Later
# versions live under their own names (syn[<n>]v<V>_<k>, module docstring); the version enters the seed and the
# structure hash, hence every pool's recipe_id.
SYN_VERSION = 1
SYN_VERSIONS = (1, 2, 3, 4, 5, 6)
_SALT = 0x5E17A                      # seed namespace of synthetic processes
_NAME = re.compile(r"^syn([2-6])?(?:v([2-6])(r2|t)?)?_(\d+)$")
DEFAULT_DIAGRAMS_DIR = os.path.join(_HERE, "data", "diagrams")

ALPHA_S, ALPHA_EW, MZ_REF = 0.118, 1.0 / 132.507, 91.1876      # alpha_s(M_Z), running reference
SQRTS_MAX = 1000.0
_COL = {k: PARTICLE_FEATURE_NAMES.index(k) for k in
        ("charge", "log10_mass_gev", "baryon_number", "lepton_number", "color_casimir")}
_NEUTRINOS = frozenset((12, 14, 16))

# initial pairs (row 0 along +z) and final-state particles with draw weights; the catalog's
# initial states are e-e+, u u~, u d~, d d~ and quark-quark
_INITIAL = [((11, -11), 0.35), ((2, -2), 0.30), ((2, -1), 0.15), ((1, -1), 0.12), ((2, 2), 0.08)]
_FINAL = [(11, 1.0), (-11, 1.0), (13, 1.0), (-13, 1.0), (12, 0.4), (-12, 0.4), (14, 0.4), (-14, 0.4),
          (1, 1.0), (-1, 1.0), (2, 1.0), (-2, 1.0), (3, 0.6), (-3, 0.6), (4, 0.6), (-4, 0.6),
          (5, 0.7), (-5, 0.7), (6, 0.5), (-6, 0.5), (21, 2.5), (22, 1.2), (23, 0.7),
          (24, 0.6), (-24, 0.6), (25, 0.4)]
_N_FINAL_P = {2: 0.3, 3: 0.35, 4: 0.35}
_MAX_FINAL_MASS = 600.0               # keeps the sqrt(s) window [~1.05 threshold, 1000] wide
# propagator particle by the channel's outgoing charge in thirds (massless first)
_PROP = {0: [(22, 0.3), (21, 0.3), (23, 0.2), (25, 0.2)],
         3: [(24, 0.5), (-11, 0.25), (-13, 0.25)], -3: [(-24, 0.5), (11, 0.25), (13, 0.25)],
         2: [(2, 0.6), (6, 0.4)], -2: [(-2, 0.6), (-6, 0.4)],
         -1: [(1, 0.6), (5, 0.4)], 1: [(-1, 0.6), (-5, 0.4)]}
_PROP_HEAVY = {0: [23, 25], 3: [24], -3: [-24], 2: [6], -2: [-6]}   # none for charge 1/3 (b is light)
_WIDTH_RATIO = (1e-3, 0.1)            # Gamma/M, log-uniform per propagator particle
_SCALAR_MASS = (20.0, 700.0)          # random scalar mass (log-uniform), half of the processes
# diagrams per process, at most. N = 7, 8 (2->5, 2->6; reachable only as syn5_<k>, syn6_<k>) added 2026-10-09 for the
# high-multiplicity test (D20), continuing the hand-set rule; every name drawable before draws exactly as before, so
# SYN_VERSION stays 1 and no existing pool changes
_KMAX = {4: 3, 5: 5, 6: 6, 7: 7, 8: 8}
_H = (2, 4)                           # helicity amplitudes summed incoherently (no exact zeros)
_LIGHT = 10.0                         # propagators below this mass (0, b) may enter as 1/sqrt


def parse_name_full(name):
    """(n_forced, version, revision, target_shaped, index) of a synthetic name (module docstring, Names)."""
    m = _NAME.match(str(name))
    if not m:
        raise KeyError(f"{name!r} is not a synthetic process name (syn[<n>][v<V>[r2|t]]_<k>)")
    n_forced = int(m.group(1)) if m.group(1) else 0
    ver, tag, idx = (int(m.group(2)) if m.group(2) else 1), m.group(3), int(m.group(4))
    if tag == "r2" and not 2 <= ver <= 5:
        raise KeyError(f"{name!r}: the r2 revision exists for versions 2-5 (version 6 is revision 2 already)")
    if tag == "t" and ver != 6:
        raise KeyError(f"{name!r}: the target-shaped tag t exists for version 6 only (version 5 is target-shaped)")
    rev = 2 if (tag == "r2" or ver == 6) else 1
    target = ver == 5 or tag == "t"
    if target and n_forced not in (0, 5, 6):
        raise KeyError(f"{name!r}: the target shape is q qbar -> V + 4g or V + 5g (n = 5, 6 or drawn)")
    return n_forced, ver, rev, target, idx


def parse_name(name):
    """(n_forced, version, index) of a synthetic name; n_forced = 0 when the multiplicity is drawn."""
    n_forced, ver, _rev, _target, idx = parse_name_full(name)
    return n_forced, ver, idx


def is_synthetic(name):
    try:
        parse_name(name)
        return True
    except KeyError:
        return False


def _props(pdg):
    p = PARTICLE_PROPERTIES.get(int(pdg))
    if p is None:
        raise KeyError(f"PDG {pdg} not in particle_ids.PARTICLE_PROPERTIES")
    return p


def leg_mass(pdg):
    """Generator mass (GeV) of a particle, from the property table (massless sentinel -> 0)."""
    lm = _props(pdg)[_COL["log10_mass_gev"]]
    # rounded back to the generator's literal value (4.7, not 4.700000000000001)
    return 0.0 if abs(lm - _MASSLESS) < 1e-6 else float(f"{10.0 ** lm:.10g}")


def _q3(pdg):
    return int(round(3 * _props(pdg)[_COL["charge"]]))


def _pick(rng, opts):
    items, w = zip(*opts)
    w = np.asarray(w, float)
    return items[int(rng.choice(len(items), p=w / w.sum()))]


def _draw_legs(rng, n):
    """Initial pair + n final particles conserving charge, baryon and lepton number."""
    ini = _pick(rng, _INITIAL)
    qn = lambda pdgs, c: round(sum(_props(p)[_COL[c]] for p in pdgs) * 3)
    order = {p: i for i, (p, _) in enumerate(_FINAL)}
    for _ in range(100000):
        fin = [_pick(rng, _FINAL) for _ in range(n)]
        if all(qn(fin, c) == qn(ini, c) for c in ("charge", "baryon_number", "lepton_number")) \
                and sum(leg_mass(p) for p in fin) <= _MAX_FINAL_MASS:
            return list(ini), sorted(fin, key=lambda p: order[p])   # identical legs adjacent
    raise RuntimeError("no conserving final state drawn")


def _random_tree(rng, N):
    """Random cubic tree over N legs: merge two random clusters until three remain (they
    meet at the root vertex). Returns the merges [(a, b)] over cluster ids (legs 0..N-1,
    merged clusters N, N+1, ...) and each cluster's leg set."""
    legs = {i: frozenset([i]) for i in range(N)}
    alive, merges = list(range(N)), []
    while len(alive) > 3:
        a, b = sorted(rng.choice(len(alive), size=2, replace=False))
        ca, cb = alive[a], alive[b]
        new = N + len(merges)
        merges.append((ca, cb))
        legs[new] = legs[ca] | legs[cb]
        alive = [c for c in alive if c not in (ca, cb)] + [new]
    return merges, legs, alive


def _canonical(S, N):
    """A channel and its complement give the same invariant: keep the side without leg 0."""
    S = frozenset(S)
    return tuple(sorted(frozenset(range(N)) - S if 0 in S else S))


@functools.lru_cache(maxsize=None)
def structure(name):
    """The random structure of synthetic process `name` (a JSON-able dict), drawn from a
    seed fixed by the name (its version included)."""
    n_forced, ver, rev, target, idx = parse_name_full(name)
    if ver == 1:
        return _structure_v1(name, n_forced, idx)
    return _structure_vk(name, ver, n_forced, idx, rev, target)


def _structure_v1(name, n_forced, idx):
    """Version 1, unchanged since it was run (every v1 pool depends on its exact draw)."""
    rng = np.random.default_rng(np.random.SeedSequence([_SALT, SYN_VERSION, n_forced, idx]))
    n = n_forced or int(_pick(rng, list(_N_FINAL_P.items())))
    ini, fin = _draw_legs(rng, n)
    pdgs = ini + fin
    N = n + 2
    m_legs = [leg_mass(p) for p in pdgs]
    sign = [-1] * 2 + [1] * n                                    # all-outgoing
    cut = [i >= 2 and m_legs[i] == 0.0 and abs(pdgs[i]) not in _NEUTRINOS for i in range(N)]
    guard = [cut[i] or m_legs[i] > 0.0 for i in range(N)]       # keeps an invariant off zero

    def protected(S):
        """A massless or light (M <= _LIGHT) propagator only where the fiducial cuts (or a leg
        mass) keep s_S off 0 (and off a light pole): s-type needs two cut legs or legs
        with masses above _LIGHT (s_S >= 100 GeV^2 either way); t-type (s_S <= 0) a guard on both
        sides of the beams."""
        if 1 not in S:
            return sum(cut[i] for i in S) >= 2 or sum(m_legs[i] for i in S) > _LIGHT
        G = [i for i in S if i >= 2]
        F = [i for i in range(2, N) if i not in S]
        return any(guard[i] for i in G) and any(guard[i] for i in F)

    colored = [_props(p)[_COL["color_casimir"]] > 0 for p in pdgs]
    n_glu = sum(p == 21 for p in fin)
    k_s = min(n, n_glu + int(rng.binomial(n - n_glu, 0.3))) if any(colored) else 0
    # a random scalar mass only without an external H: one PDG must not carry two masses
    m_scalar = float(np.exp(rng.uniform(*np.log(_SCALAR_MASS)))) if rng.random() >= 0.5 else 125.0
    if 25 in fin:
        m_scalar = 125.0
    ratio = {}

    def width(pdg):
        a = abs(pdg)
        if a not in ratio:
            ratio[a] = float(np.exp(rng.uniform(*np.log(_WIDTH_RATIO))))
        return ratio[a]

    diagrams, seen = [], set()
    n_diag = int(rng.integers(1, _KMAX[N] + 1))
    for _ in range(200 * n_diag):
        if len(diagrams) == n_diag:
            break
        merges, legs_of, root = _random_tree(rng, N)
        props, ok = [], True
        for c in range(N, N + len(merges)):
            S = _canonical(legs_of[c], N)
            q = sum(sign[i] * _q3(pdgs[i]) for i in S)
            if q not in _PROP:
                ok = False; break
            pdg = int(_pick(rng, _PROP[q]))
            if leg_mass(pdg) <= _LIGHT and not protected(S):
                if q not in _PROP_HEAVY:            # charge 1/3: no heavy option, redraw the tree
                    ok = False; break
                pdg = int(_PROP_HEAVY[q][int(rng.integers(len(_PROP_HEAVY[q])))])
            M = m_scalar if abs(pdg) == 25 else leg_mass(pdg)
            s_type = 1 not in S
            props.append({"legs": list(S), "pdg": pdg, "mass": M,
                          "width": (width(pdg) * M if (s_type and M > 0) else 0.0),
                          "power": (1.0 if M > _LIGHT else float(rng.choice([0.5, 1.0])))})
        if not ok:
            continue
        key = frozenset((tuple(p["legs"]), p["pdg"]) for p in props)
        if key in seen:
            continue
        seen.add(key)
        qcd = sorted(rng.choice(N - 2, size=k_s, replace=False).tolist())   # QCD vertices
        diagrams.append({"merges": [list(mm) for mm in merges], "root": [int(r) for r in root],
                         "props": props, "qcd_vertices": qcd})
    if not diagrams:
        raise RuntimeError(f"{name}: no diagram with SM-representable channels")

    pairs = list(itertools.combinations(range(N), 2))
    H = int(rng.integers(_H[0], _H[1] + 1))
    numer = []
    for _h in range(H):
        row = []
        for d in diagrams:
            deg = int(rng.integers(0, 3))
            monos = [{"a": float(rng.normal()),
                      "pairs": sorted(int(x) for x in rng.choice(len(pairs), size=deg))}
                     for _ in range(int(rng.integers(1, 4)))]
            e_sum = sum(p["power"] for p in d["props"])
            row.append({"c": [float(rng.normal()), float(rng.normal())], "deg": deg, "monos": monos,
                        # s-power giving the term mass dimension 4 - N: 2 deg - k - 2 e_sum = 4 - N
                        "k": 2 * deg - 2 * e_sum - 4 + N})
        numer.append(row)

    thr = sum(m_legs[2:])
    lo = max(25.0, 1.05 * thr)                                 # the catalog's window rule
    st = {
        "name": name, "version": SYN_VERSION, "pdg_ids": pdgs, "m_legs": m_legs, "nfinal": n,
        "sqrts": [float(np.ceil(lo)), SQRTS_MAX], "k_s": int(k_s),
        "coupling": float((4 * np.pi) ** n * ALPHA_EW ** (n - k_s)),   # times alpha_s(sqrt s)^k_s
        "internal_masses": {25: m_scalar}, "diagrams": diagrams, "pairs": [list(p) for p in pairs],
        "numerators": numer, "perms": _identical_perms(pdgs),
    }
    st["sha"] = hashlib.sha256(json.dumps(st, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return st


def _identical_perms(pdgs):
    """Every relabelling of the slots that only exchanges identical particles (initial and
    final kept apart): the group the target is averaged over."""
    N = len(pdgs)
    groups = {}
    for i, p in enumerate(pdgs):
        groups.setdefault((i < 2, p), []).append(i)
    blocks = [g for g in groups.values() if len(g) > 1]
    out = []
    for choice in itertools.product(*[list(itertools.permutations(g)) for g in blocks]):
        perm = list(range(N))
        for g, img in zip(blocks, choice):
            for a, b in zip(g, img):
                perm[a] = b
        out.append(perm)
    return out


def _mink2(v):
    return v[..., 0] ** 2 - (v[..., 1:] ** 2).sum(-1)


def evaluate(st, P, scale=1.0):
    """|M|^2 of structure `st` at momenta P (B, N, 4) (rows in pdg_ids order, beams incoming).
    `scale` multiplies every dimensionful constant (propagator masses and widths): the
    mass-dimension check is |M|^2(lam P, scale=lam) = lam^(8-2N) |M|^2(P).
    Version 1 keeps its original evaluator (bit-identical labels); later versions use the
    vectorised one, which gives v1 structures the same values to rounding (tests/test_synthetic_versions.py)."""
    if st.get("version", 1) == 1:
        return _evaluate_v1(st, P, scale)
    if st["version"] >= 6:
        return evaluate_v6(st, P, scale)
    return evaluate_vec(st, P, scale)


def _alpha_s(P, scale):
    """alpha_s(sqrt s) per event, one loop from M_Z (nf = 5) as mg5_pipeline_final.compute_alphas."""
    b0 = (11 * 3 - 2 * 5) / (12 * np.pi)
    s_in = _mink2(P[:, 0] + P[:, 1])
    return ALPHA_S / (1 + ALPHA_S * b0 * np.log(s_in / (MZ_REF * scale) ** 2))


def _evaluate_v1(st, P, scale=1.0):
    P = np.asarray(P, np.float64)
    N = P.shape[1]
    sign = np.array([-1.0, -1.0] + [1.0] * (N - 2))
    pairs = [tuple(p) for p in st["pairs"]]
    total = np.zeros(P.shape[0])
    for perm in st["perms"]:
        q = P[:, perm, :] * sign[None, :, None]
        s_tot = _mink2(q[:, 0] + q[:, 1])
        s_pair = np.stack([_mink2(q[:, i] + q[:, j]) for i, j in pairs], axis=1)
        cache = {}
        dens = []
        for d in st["diagrams"]:
            den = np.ones(P.shape[0], np.complex128)
            for pr in d["props"]:
                S = tuple(pr["legs"])
                if S not in cache:
                    cache[S] = _mink2(q[:, list(S)].sum(1))
                sS = cache[S]
                M, G = pr["mass"] * scale, pr["width"] * scale
                D = sS - M * M + 1j * M * G
                if pr["power"] == 1.0:
                    den = den * D
                elif M > 0:
                    den = den * np.sqrt(D)          # principal branch (cut only where D <= 0, G = 0)
                else:
                    den = den * np.sqrt(np.abs(sS))
            dens.append(den)
        for row in st["numerators"]:
            A = np.zeros(P.shape[0], np.complex128)
            for term, den in zip(row, dens):
                poly = np.zeros(P.shape[0])
                for mono in term["monos"]:
                    t = np.full(P.shape[0], mono["a"])
                    for k in mono["pairs"]:
                        t = t * s_pair[:, k]
                    poly = poly + t
                A = A + complex(*term["c"]) * poly * s_tot ** (-0.5 * term["k"]) / den
            total = total + np.abs(A) ** 2
    # alpha_s runs per event at mu = sqrt(s) as in the real pools (mg5_pipeline_final.compute_alphas,
    # one loop from M_Z, nf = 5); M_Z scales with the other constants
    b0 = (11 * 3 - 2 * 5) / (12 * np.pi)
    s_in = _mink2(P[:, 0] + P[:, 1])
    a_s = ALPHA_S / (1 + ALPHA_S * b0 * np.log(s_in / (MZ_REF * scale) ** 2))
    return st["coupling"] * a_s ** st["k_s"] * total / len(st["perms"])


# =============================================================================
# Versions 2-5 (site decisions D36, D38): cumulative changes on version 1, each under its own names
# (syn[<n>]v<V>[r2]_<k>); the module docstring states what each adds.
# =============================================================================
_KFRAC = 0.25                         # v2+: K |Pi| <= _KFRAC (2N-5)!!, K log-uniform (module docstring)
_N_FINAL_P5 = {5: 0.5, 6: 0.5}        # v5: V + 4g or V + 5g
_V5_BOSONS = (22, 23, 24, 25)         # v5: V by the index, idx % 4 (any four consecutive names hold all four)
_SELF_CONJ = frozenset((21, 22, 23, 25))
_FERMIONS = (1, 2, 3, 4, 5, 6, 11, 12, 13, 14)
_DOUBLETS = ((2, 1), (4, 3), (6, 5), (12, 11), (14, 13))       # (up, down) of each W vertex: diagonal CKM
_YUKAWA = (5, 6)                      # fermions with a generator mass: the only f fbar H vertices
_FAMILY = {1: "q1", 2: "q1", 3: "q2", 4: "q2", 5: "q3", 6: "q3", 11: "le", 12: "le", 13: "lm", 14: "lm"}
_GAUGE = frozenset((21, 22))          # massless gauge bosons: the legs that can go soft
# v3+: line particle weights among the legal options (version 1's charge-table weights where it had them)
_LINE_W = {21: 0.3, 22: 0.3, 23: 0.2, 25: 0.2, 24: 0.5, -24: 0.5}
_LINE_W.update({s * f: (0.25 if f > 10 else (0.4 if f in (5, 6) else 0.6)) for f in _FERMIONS for s in (1, -1)})
_ORDER_OF = {"QED": {"QED": 1}, "QCD": {"QCD": 1}, "HGG": {"QCD": 2, "QED": 1}}


def _anti(p):
    return p if p in _SELF_CONJ else -p


def _double_factorial(m):
    return int(np.prod(np.arange(m, 0, -2))) if m > 0 else 1


@functools.lru_cache(maxsize=None)
def sm_vertices(hgg=False):
    """The SM cubic vertices as sorted all-outgoing pdg triples -> coupling class: f fbar gamma (charged f),
    f fbar Z, q qbar g, f fbar H (b, t: the massive fermions of the generator), the W vertices of the doublets
    (diagonal CKM, lepton families), W W gamma, W W Z, g g g, W W H, Z Z H; with `hgg` (version 5) the effective
    g g H vertex (heavy-top limit), class 'HGG'."""
    V = {}

    def add(t, cls):
        V[tuple(sorted(t))] = cls
    for f in _FERMIONS:
        if _q3(f) != 0:
            add((f, -f, 22), "QED")
        add((f, -f, 23), "QED")
        if f <= 6:
            add((f, -f, 21), "QCD")
        if f in _YUKAWA:
            add((f, -f, 25), "QED")
    for up, dn in _DOUBLETS:
        add((-up, dn, 24), "QED")
        add((up, -dn, -24), "QED")
    for t, cls in (((24, -24, 22), "QED"), ((24, -24, 23), "QED"), ((21, 21, 21), "QCD"),
                   ((24, -24, 25), "QED"), ((23, 23, 25), "QED")):
        add(t, cls)
    if hgg:
        add((21, 21, 25), "HGG")
    return V


@functools.lru_cache(maxsize=None)
def _line_table(hgg):
    """(Xa, Xb) -> [(X, class)]: the lines X that a vertex joining the outgoing lines Xa, Xb can carry on, i.e.
    (Xa, Xb, anti X) is an SM vertex. X is the line's particle seen as the merged cluster's outgoing state."""
    V = sm_vertices(hgg)
    univ = sorted({p for t in V for p in t})
    tab = {}
    for a in univ:
        for b in univ:
            tab[(a, b)] = [(X, V[tuple(sorted((a, b, _anti(X))))]) for X in univ
                           if tuple(sorted((a, b, _anti(X)))) in V]
    return tab


def vertex_list(st, d):
    """Every vertex of diagram d as (all-outgoing pdg triple, [is each line an external leg]), rebuilt from its
    merges: the check of vertex legality reads this, not the builder's own bookkeeping."""
    pdgs = st["pdg_ids"]
    N = len(pdgs)
    X = {i: (_anti(pdgs[i]) if i < 2 else pdgs[i]) for i in range(N)}
    for c, pr in zip(range(N, N + len(d["merges"])), d["props"]):
        X[c] = pr["x"]
    out = []
    for c, (a, b) in zip(range(N, N + len(d["merges"])), d["merges"]):
        out.append(((X[a], X[b], _anti(X[c])), [a < N, b < N, False]))
    out.append((tuple(X[c] for c in d["root"]), [c < N for c in d["root"]]))
    return out


def _families(pdgs):
    out = {}
    for p in pdgs:
        f = _FAMILY.get(abs(p))
        if f:
            out[f] = out.get(f, 0) + (1 if p > 0 else -1)
    return {k: v for k, v in out.items() if v}


def _draw_legs_vk(rng, n, flavour):
    """Version 1's leg draw; with `flavour` (v3+) also quark-generation and lepton-family numbers conserved
    (diagonal CKM, massless neutrinos: the SM at tree level)."""
    ini = _pick(rng, _INITIAL)
    qn = lambda pdgs, c: round(sum(_props(p)[_COL[c]] for p in pdgs) * 3)
    order = {p: i for i, (p, _) in enumerate(_FINAL)}
    for _ in range(100000):
        fin = [_pick(rng, _FINAL) for _ in range(n)]
        if all(qn(fin, c) == qn(ini, c) for c in ("charge", "baryon_number", "lepton_number")) \
                and sum(leg_mass(p) for p in fin) <= _MAX_FINAL_MASS \
                and (not flavour or _families(fin) == _families(ini)):
            return list(ini), sorted(fin, key=lambda p: order[p])
    raise RuntimeError("no conserving final state drawn")


def _draw_legs_v5(rng, n, idx):
    """q qbar' -> V + (n-1) g: V = gamma, Z, W+, H by idx % 4; u ubar or d dbar (u dbar for the W, as the catalog's
    udbar_W* processes)."""
    V = _V5_BOSONS[idx % len(_V5_BOSONS)]
    ini = (2, -1) if V == 24 else _pick(rng, [((2, -2), 0.5), ((1, -1), 0.5)])
    return list(ini), [V] + [21] * (n - 1)


def _orbit_key(props, perms, N):
    """A diagram up to the identical-leg relabellings (the permutation average already sums its images)."""
    return min(tuple(sorted((_canonical([perm[i] for i in p["legs"]], N), abs(p["pdg"])) for p in props))
               for perm in perms)


def _tree_charge(rng, N, pdgs, sign, protected):
    """Version 1's tree: a random cubic tree, each line drawn from the channel-charge table (v2)."""
    merges, legs_of, root = _random_tree(rng, N)
    props = []
    for c in range(N, N + len(merges)):
        S = _canonical(legs_of[c], N)
        q = sum(sign[i] * _q3(pdgs[i]) for i in S)
        if q not in _PROP:
            return None
        pdg = int(_pick(rng, _PROP[q]))
        if leg_mass(pdg) <= _LIGHT and not protected(S):
            if q not in _PROP_HEAVY:
                return None
            pdg = int(_PROP_HEAVY[q][int(rng.integers(len(_PROP_HEAVY[q])))])
        props.append({"legs": list(S), "pdg": pdg})
    return {"merges": [list(mm) for mm in merges], "root": [int(r) for r in root], "props": props}


def _tree_legal(rng, N, out, protected, mass, hgg):
    """A cubic tree whose every vertex is an SM vertex (v3+). Built bottom-up: at each step a pair of clusters is
    chosen uniformly among those some legal line can join (after the light-line protection), the line's particle
    among those legal lines by _LINE_W; the three clusters left must meet at a legal vertex, else the tree is
    rejected. The effective g g H vertex (v5) is legal only on the external H."""
    tab, V = _line_table(hgg), sm_vertices(hgg)
    cl = {i: (frozenset([i]), out[i], True) for i in range(N)}
    alive, merges, props, vcls = list(range(N)), [], [], []
    while len(alive) > 3:
        cands = []
        for x in range(len(alive)):
            for y in range(x + 1, len(alive)):
                ca, cb = alive[x], alive[y]
                (La, Xa, la), (Lb, Xb, lb) = cl[ca], cl[cb]
                opts = [(X, c) for X, c in tab.get((Xa, Xb), ())
                        if c != "HGG" or (Xa == 25 and la) or (Xb == 25 and lb)]
                if not opts:
                    continue
                S = _canonical(La | Lb, N)
                if any(mass(X) <= _LIGHT for X, _ in opts) and not protected(S):
                    opts = [(X, c) for X, c in opts if mass(X) > _LIGHT]
                if opts:
                    cands.append((ca, cb, S, opts))
        if not cands:
            return None
        ca, cb, S, opts = cands[int(rng.integers(len(cands)))]
        w = np.array([_LINE_W[X] for X, _ in opts], float)
        X, cls = opts[int(rng.choice(len(opts), p=w / w.sum()))]
        new = N + len(merges)
        merges.append((ca, cb))
        vcls.append(cls)
        L = cl[ca][0] | cl[cb][0]
        cl[new] = (L, X, False)
        # pdg of the line as seen from the stored (canonical) side, as version 1 stores it
        props.append({"legs": list(S), "pdg": int(X if tuple(sorted(L)) == S else _anti(X)), "x": int(X)})
        alive = [c for c in alive if c not in (ca, cb)] + [new]
    cls = V.get(tuple(sorted(cl[c][1] for c in alive)))
    if cls is None or (cls == "HGG" and not any(cl[c][1] == 25 and cl[c][2] for c in alive)):
        return None
    vcls.append(cls)
    return {"merges": [list(mm) for mm in merges], "root": [int(c) for c in alive], "props": props,
            "vertex_class": vcls}


def _structure_vk(name, ver, n_forced, idx, rev=1, target=None):
    # revision 1 (v2-v5 as first run) keeps its seed and code path byte for byte; revision 2 (r2 names, v6) has
    # its own seed, so no r2 name can reproduce or collide with a revision-1 pool
    target = ver >= 5 if target is None else target
    ent = [_SALT, ver, n_forced, idx] if rev == 1 else [_SALT, ver, n_forced, idx, rev, int(target)]
    rng = np.random.default_rng(np.random.SeedSequence(ent))
    n = n_forced or int(_pick(rng, list((_N_FINAL_P5 if target else _N_FINAL_P).items())))
    for _ in range(1000):
        ini, fin = _draw_legs_v5(rng, n, idx) if target else _draw_legs_vk(rng, n, flavour=ver >= 3)
        if rev == 1:
            st = _draw_process_vk(rng, name, ver, ini, fin)
        elif ver == 2:
            st = _draw_process_vk(rng, name, ver, ini, fin, fixed_k=True)
        else:
            st = _draw_process_r2(rng, name, ver, ini, fin, target)
        if st is not None:
            return st
    raise RuntimeError(f"{name}: no final state with a diagram drawn")


def _draw_process_vk(rng, name, ver, ini, fin, fixed_k=False):
    """Everything after the legs, versions 2-5; None when no diagram can be built on these legs (v3+: no SM tree)."""
    pdgs = ini + fin
    n, N = len(fin), len(pdgs)
    m_legs = [leg_mass(p) for p in pdgs]
    sign = [-1] * 2 + [1] * n
    cut = [i >= 2 and m_legs[i] == 0.0 and abs(pdgs[i]) not in _NEUTRINOS for i in range(N)]
    guard = [cut[i] or m_legs[i] > 0.0 for i in range(N)]

    def protected(S):                                       # version 1's rule
        if 1 not in S:
            return sum(cut[i] for i in S) >= 2 or sum(m_legs[i] for i in S) > _LIGHT
        G = [i for i in S if i >= 2]
        F = [i for i in range(2, N) if i not in S]
        return any(guard[i] for i in G) and any(guard[i] for i in F)

    m_scalar = float(np.exp(rng.uniform(*np.log(_SCALAR_MASS)))) if rng.random() >= 0.5 else 125.0
    if 25 in fin:
        m_scalar = 125.0
    mass = lambda X: m_scalar if abs(X) == 25 else leg_mass(X)
    ratio = {}

    def width(pdg):
        a = abs(pdg)
        if a not in ratio:
            ratio[a] = float(np.exp(rng.uniform(*np.log(_WIDTH_RATIO))))
        return ratio[a]

    perms = _identical_perms(pdgs)
    k_cap = max(_KMAX[N], int(_KFRAC * _double_factorial(2 * N - 5)) // len(perms))
    if fixed_k:                                             # v2r2: K = the cap (module docstring)
        n_diag = int(k_cap)
    else:
        n_diag = int(min(k_cap, max(1, np.floor(np.exp(rng.uniform(0.0, np.log(k_cap + 1)))))))
    out = [_anti(p) for p in ini] + list(fin)              # all-outgoing particles
    diagrams, seen = [], set()
    for _ in range(50 * n_diag + 2000):
        if len(diagrams) == n_diag:
            break
        d = _tree_charge(rng, N, pdgs, sign, protected) if ver == 2 else \
            _tree_legal(rng, N, out, protected, mass, hgg=ver >= 5)
        if d is None:
            continue
        key = _orbit_key(d["props"], perms, N)
        if key in seen:
            continue
        seen.add(key)
        spinors = []
        for pr in d["props"]:
            S, M = pr["legs"], mass(pr["pdg"])
            s_type = 1 not in S
            pr["mass"] = M
            pr["width"] = width(pr["pdg"]) * M if (s_type and M > 0) else 0.0
            if ver >= 4 and M == 0.0 and s_type and len(S) == 2 and all(m_legs[i] == 0.0 for i in S):
                pr["power"] = 1.0                           # v4: the numerator vanishes instead
                spinors.append({"legs": list(S), "soft": [i for i in S if pdgs[i] in _GAUGE]})
            else:
                pr["power"] = 1.0 if M > _LIGHT else float(rng.choice([0.5, 1.0]))
        if ver >= 4:
            d["spinors"] = spinors
        diagrams.append(d)
    if not diagrams:
        return None

    if ver == 2:                                            # version 1's drawn QCD order
        colored = [_props(p)[_COL["color_casimir"]] > 0 for p in pdgs]
        n_glu = sum(p == 21 for p in fin)
        k_s = min(n, n_glu + int(rng.binomial(n - n_glu, 0.3))) if any(colored) else 0
        for d in diagrams:
            d["qcd_vertices"] = sorted(rng.choice(N - 2, size=k_s, replace=False).tolist())
        coupling, order = float((4 * np.pi) ** n * ALPHA_EW ** (n - k_s)), [0, 0, int(k_s), int(n - k_s)]
    else:                                                   # couplings from the vertices, per diagram
        e = np.sqrt(4 * np.pi * ALPHA_EW)
        for d in diagrams:
            nq, ne, nh = (sum(c == k for c in d["vertex_class"]) for k in ("QCD", "QED", "HGG"))
            d["as_pow"] = 0.5 * nq + nh                     # amplitude ~ g_s^nq e^ne (alpha_s e / 3 pi)^nh
            d["cpl"] = float((4 * np.pi) ** (0.5 * nq) * e ** ne * (e / (3 * np.pi)) ** nh)
            d["n_ew"] = ne + nh
        # the order vector holds the maxima over the diagrams, as mg5_pipeline_final.order_vector (a mixed
        # four-quark entry is [0, 0, 2, 2]); alphas_power is the alpha_s maximum
        k_s = int(round(2 * max(d["as_pow"] for d in diagrams)))
        coupling, order = 1.0, [0, 0, k_s, int(max(d["n_ew"] for d in diagrams))]

    pairs = list(itertools.combinations(range(N), 2))
    H = int(rng.integers(_H[0], _H[1] + 1))
    numer = []
    for _h in range(H):
        row = []
        for d in diagrams:
            sp = d.get("spinors", [])
            banned = {tuple(s["legs"]) for s in sp}
            allowed = [i for i, p in enumerate(pairs) if p not in banned]
            deg = int(rng.integers(0, 3))
            monos = [{"a": float(rng.normal()),
                      "pairs": sorted(int(allowed[x]) for x in rng.choice(len(allowed), size=deg))}
                     for _ in range(int(rng.integers(1, 4)))]
            e_sum = sum(p["power"] for p in d["props"])
            term = {"c": [float(rng.normal()), float(rng.normal())], "deg": deg, "monos": monos,
                    # mass dimension 4 - N: 2 deg + n_brackets - k - 2 e_sum = 4 - N
                    "k": 2 * deg + len(sp) - 2 * e_sum - 4 + N}
            if sp:
                term["sig"] = [int(x) for x in rng.choice([-1, 1], size=len(sp))]   # <ij> (+1) or [ij] (-1)
            row.append(term)
        numer.append(row)

    thr = sum(m_legs[2:])
    lo = max(25.0, 1.05 * thr)
    st = {
        "name": name, "version": ver, "pdg_ids": pdgs, "m_legs": m_legs, "nfinal": n,
        "sqrts": [float(np.ceil(lo)), SQRTS_MAX], "k_s": int(k_s), "coupling": coupling,
        "coupling_mode": "global" if ver == 2 else "vertex", "order": order, "k_cap": int(k_cap),
        "internal_masses": {25: m_scalar}, "diagrams": diagrams, "pairs": [list(p) for p in pairs],
        "numerators": numer, "perms": perms,
    }
    if fixed_k:
        st["revision"], st["k_drawn"] = 2, len(diagrams)
    st["sha"] = hashlib.sha256(json.dumps(st, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return st


_PLANS = {}


def _plan(st):
    """Index arrays of the vectorised evaluator: every (permutation, diagram) image's propagators, monomials,
    brackets and couplings as gathers into per-event tables (channels, pair invariants, brackets)."""
    pl = _PLANS.get(st["sha"])
    if pl is not None:
        return pl
    N, perms, D = len(st["pdg_ids"]), st["perms"], st["diagrams"]
    rows = st["numerators"] if st["version"] < 6 else []     # v6: helicity configurations, _plan6
    K, H, nP = len(D), len(rows), len(perms)
    T = nP * K
    pairs = [tuple(p) for p in st["pairs"]]
    pid = {p: i for i, p in enumerate(pairs)}
    ONE = len(pairs)
    nsp = max((len(d.get("spinors", ())) for d in D), default=0)
    chans, ptypes, brs = {}, {}, {}
    den = np.zeros((T, N - 3), np.int64)
    bidx = -np.ones((T, max(nsp, 1)), np.int64)
    m1 = np.full((T, H, 3), ONE, np.int64)
    m2 = np.full((T, H, 3), ONE, np.int64)
    a = np.zeros((T, H, 3))
    c = np.zeros((T, H), np.complex128)
    kk = np.zeros((T, H))
    sig = np.ones((T, H, max(nsp, 1)), np.int64)
    const = np.ones(T)
    as_pow = np.zeros(T)
    for s, perm in enumerate(perms):
        pmap = [pid[tuple(sorted((perm[i], perm[j])))] for i, j in pairs]
        for di, d in enumerate(D):
            t = s * K + di
            # w = 1/|stabiliser| of the orbit representative (revision 2's coherent orbit sum), 1 otherwise
            const[t], as_pow[t] = d.get("cpl", 1.0) * d.get("w", 1.0), d.get("as_pow", 0.0)
            for j, pr in enumerate(d["props"]):
                S = _canonical([perm[i] for i in pr["legs"]], N)
                ch = chans.setdefault(S, len(chans))
                den[t, j] = ptypes.setdefault((ch, pr["mass"], pr["width"], pr["power"]), len(ptypes))
            for j, spn in enumerate(d.get("spinors", ())):
                i, k = spn["legs"]
                key = (perm[i], perm[k], perm[0], perm[1], tuple(sorted(perm[x] for x in spn["soft"])))
                bidx[t, j] = brs.setdefault(key, len(brs))
            for h, row in enumerate(rows):
                term = row[di]
                c[t, h], kk[t, h] = complex(*term["c"]), term["k"]
                for m, mono in enumerate(term["monos"]):
                    ps = [pmap[x] for x in mono["pairs"]] + [ONE, ONE]
                    m1[t, h, m], m2[t, h, m], a[t, h, m] = ps[0], ps[1], mono["a"]
                for j, sg in enumerate(term.get("sig", ())):
                    sig[t, h, j] = sg
    bidx[bidx < 0] = len(brs)                              # padding: the ones column
    chmask = np.zeros((len(chans), N))
    for S, ch in chans.items():
        chmask[ch, list(S)] = 1.0
    pt = sorted(ptypes.items(), key=lambda kv: kv[1])
    ku, kinv = np.unique(kk, return_inverse=True)
    mq, qidx = np.unique(m1 * (ONE + 1) + m2, return_inverse=True)
    br = sorted(brs.items(), key=lambda kv: kv[1])
    soft = np.zeros((len(br), N))
    for (key, b) in br:
        soft[b, list(key[4])] = 1.0
    pl = {"T": T, "K": K, "H": H, "nP": nP, "nsp": nsp, "den": den, "chmask": chmask,
          "ptC": np.array([k[0] for k, _ in pt], np.int64), "ptM": np.array([k[1] for k, _ in pt]),
          "ptW": np.array([k[2] for k, _ in pt]), "ptE": np.array([k[3] for k, _ in pt]),
          "pairs": np.array(pairs, np.int64), "qa": mq // (ONE + 1), "qb": mq % (ONE + 1),
          "qidx": qidx.reshape(m1.shape), "a": a, "c": c,
          "ku": ku, "kinv": kinv.reshape(T, H), "const": const, "as_pow": as_pow,
          "bidx": bidx, "sig": sig, "nbr": len(br),
          "bi": np.array([k[0] for k, _ in br], np.int64), "bk": np.array([k[1] for k, _ in br], np.int64),
          "ba": np.array([k[2] for k, _ in br], np.int64), "bb": np.array([k[3] for k, _ in br], np.int64),
          "soft": soft}
    # images with the same propagators and couplings (a stabiliser's images of one labelled diagram, revision 2)
    # share one propagator product, computed once and gathered (bitwise the same values)
    key = np.concatenate([den.astype(float), const[:, None], as_pow[:, None]], 1)
    uk, grp = np.unique(key, axis=0, return_inverse=True)
    if len(uk) < 0.8 * T:
        pl["dgrp"] = grp.reshape(-1)
        pl["den_u"], pl["const_u"], pl["aspow_u"] = uk[:, :-2].astype(np.int64), uk[:, -2], uk[:, -1]
    if nsp:                          # bracket products by sign pattern relative to the first bracket (_eval_chunk)
        rel = (sig[:, :, 1:] * sig[:, :, :1]) < 0           # (T, H, nsp-1): conjugate bracket j relative to 0
        pl["pid"] = (rel * (1 << np.arange(nsp - 1))[None, None, :]).sum(-1) if nsp > 1 else np.zeros((T, H), int)
        pl["s0"] = sig[:, :, 0].astype(float)
        pl["patterns"] = sorted(set(pl["pid"].reshape(-1).tolist()))
    pl["sparse"] = _sparse_rows(pl, T, H, nsp, nP, K, bool(st.get("coherent")))
    _PLANS[st["sha"]] = pl
    return pl


def _sparse_rows(pl, T, H, nsp, nP, K, coherent):
    """The numerator sums of Eq. (syn) as sparse matrices, one per helicity row h (and per real/imaginary bracket
    part): sum_t X_t br_t = sum_rows spow_k Q_q (S_h Y)_row, rows = (permutation group, s-power k, monomial q),
    columns = (sign pattern, image t), entries c a (times the first bracket's sign for the imaginary part), Y the
    per-event propagators times the pattern's bracket product. Same terms as the elementwise sums, summed in another
    order (labels equal to rounding, tests/test_synthetic_versions.py and the before/after check)."""
    from scipy import sparse
    npat = len(pl["patterns"]) if nsp else 1
    pcol = {p: i for i, p in enumerate(pl["patterns"])} if nsp else {0: 0}
    t = np.arange(T)
    g = np.zeros(T, np.int64) if coherent else t // K
    # without brackets a column is a propagator product: images sharing one (the stabiliser's) share the column
    tcol, ncol = (pl["dgrp"], len(pl["const_u"])) if (not nsp and "dgrp" in pl) else (t, T)
    res = []
    for h in range(H):
        pidx = (np.array([pcol[x] for x in pl["pid"][:, h]]) if nsp else np.zeros(T, np.int64))
        rows_t, cols, vals_re, vals_im, keys = [], [], [], [], []
        for m in range(3):
            a = pl["a"][:, h, m]
            ok = a != 0
            q = pl["qidx"][:, h, m]
            key = (g * len(pl["ku"]) + pl["kinv"][:, h]) * len(pl["qa"]) + q
            keys.append(key[ok])
            cols.append((pidx * ncol + tcol)[ok])
            w = (pl["c"][:, h] * a)[ok]
            vals_re.append(w)
            vals_im.append(w * (pl["s0"][ok, h] if nsp else 1.0))
        key = np.concatenate(keys)
        uk, row = np.unique(key, return_inverse=True)
        col = np.concatenate(cols)
        shape = (len(uk), npat * ncol)
        S_re = sparse.csr_matrix((np.concatenate(vals_re), (row, col)), shape=shape)
        S_im = sparse.csr_matrix((np.concatenate(vals_im), (row, col)), shape=shape) if nsp else None
        rq = uk % len(pl["qa"])
        rk = (uk // len(pl["qa"])) % len(pl["ku"])
        rg = uk // (len(pl["qa"]) * len(pl["ku"]))
        res.append({"S_re": S_re, "S_im": S_im, "rq": rq, "rk": rk, "rg": rg,
                    "gstart": np.searchsorted(rg, np.arange(1 if coherent else nP))})
    return res


def _dot(x, y):
    return x[..., 0] * y[..., 0] - (x[..., 1:] * y[..., 1:]).sum(-1)


def brackets(q, i, k, a, b, soft):
    """The v4 numerator factor of a massless line on the two massless legs i, k (all-outgoing momenta q (B, N, 4);
    index arrays of equal length; `soft` (len, N) marks the gauge-boson legs among i, k):

        B_ik = |s_ik|^(1/2) * W_ik * tr(i k b a) / |tr(i k b a)|,
        tr(i k b a) = 2[(q_i.q_k)(q_b.q_a) - (q_i.q_b)(q_k.q_a) + (q_i.q_a)(q_k.q_b)] + 2i eps(q_i, q_k, q_b, q_a)
                    = <ik>[kb]<ba>[ai]  (massless; |tr|^2 = s_ik s_kb s_ba s_ai),
        W_ik = prod_{x in {i,k} soft-able} ((q_i + q_k).P / q_x.P)^(1/2),   P = p_a + p_b,

    so |B| = |<ik>| = |s_ik|^(1/2) times the energy-fraction factor, and its phase is that of <ik> with the
    little-group phases fixed by the two beams (weight 0 on every leg: Lorentz invariant, no helicity
    bookkeeping). [ik] is its complex conjugate. Returns (B, len) complex."""
    qi, qk, qa, qb = q[:, i], q[:, k], q[:, a], q[:, b]
    re = 2 * (_dot(qi, qk) * _dot(qb, qa) - _dot(qi, qb) * _dot(qk, qa) + _dot(qi, qa) * _dot(qk, qb))
    eps = np.linalg.det(np.stack([qi, qk, qb, qa], axis=-2))
    tr = re + 2j * eps
    mod = np.abs(tr)
    phase = np.where(mod > 0, tr / np.where(mod > 0, mod, 1.0), 1.0)
    P = -(q[:, 0] + q[:, 1])
    E = _dot(q, P[:, None, :])                             # sqrt(s) times each leg's CM energy
    Eik = E[:, i] + E[:, k]
    logw = 0.5 * (soft.sum(1)[None, :] * np.log(Eik) - (np.log(np.abs(E))[:, None, :] * soft[None]).sum(-1))
    return np.sqrt(np.abs(2 * _dot(qi, qk))) * np.exp(logw) * phase


def evaluate_vec(st, P, scale=1.0):
    """Vectorised evaluation of Eq. (syn) (any version; v1 structures to rounding): chunks of events, all
    permutation images of all diagrams at once."""
    pl = _plan(st)
    P = np.asarray(P, np.float64)
    out = np.empty(P.shape[0])
    step = int(min(4096, max(8, 1_000_000 // max(pl["T"] * max(pl["nsp"], 1), 1))))
    for lo in range(0, P.shape[0], step):
        out[lo:lo + step] = _eval_chunk(st, pl, P[lo:lo + step], scale)
    return out


def _eval_chunk(st, pl, P, scale):
    B, N = P.shape[:2]
    nsp, nP, K = pl["nsp"], pl["nP"], pl["K"]
    coherent = bool(st.get("coherent"))
    cs = 1 if coherent else nP                             # incoherent: |.|^2 per permutation image, averaged
    sign = np.array([-1.0, -1.0] + [1.0] * (N - 2))
    q = P * sign[None, :, None]
    s = _mink2(np.einsum("cn,bnk->bck", pl["chmask"], q))[:, pl["ptC"]]
    M, G = pl["ptM"] * scale, pl["ptW"] * scale
    val = s - M * M + 1j * M * G
    half = pl["ptE"] != 1.0
    if half.any():
        val[:, half] = np.where(M[half] > 0, np.sqrt(val[:, half]), np.sqrt(np.abs(s[:, half])))
    inv = 1.0 / val
    a_s = _alpha_s(P, scale)
    grp = pl.get("dgrp")
    den, const, aspow = ((pl["den_u"], pl["const_u"], pl["aspow_u"]) if grp is not None else
                         (pl["den"], pl["const"], pl["as_pow"]))
    dinv = inv[:, den[:, 0]]
    for j in range(1, den.shape[1]):
        dinv *= inv[:, den[:, j]]
    if st.get("coupling_mode", "global") == "vertex":
        dinv *= const[None, :] * a_s[:, None] ** aspow[None, :]
    if grp is not None and nsp:
        dinv = dinv[:, grp]
    pairs = pl["pairs"]
    SP = np.ones((B, len(pairs) + 1))
    SP[:, :-1] = _mink2(q[:, pairs[:, 0]] + q[:, pairs[:, 1]])
    Q = SP[:, pl["qa"]] * SP[:, pl["qb"]]                  # every monomial (degree <= 2) used, once
    spow = _mink2(q[:, 0] + q[:, 1])[:, None] ** (-0.5 * pl["ku"][None, :])
    if nsp:
        Bv = np.ones((B, pl["nbr"] + 1), np.complex128)
        Bv[:, :-1] = brackets(q, pl["bi"], pl["bk"], pl["ba"], pl["bb"], pl["soft"])
        bj = [Bv[:, pl["bidx"][:, j]] for j in range(nsp)]
        # the bracket product of every sign pattern, once per chunk (not once per helicity row): pattern bit j - 1
        # conjugates bracket j relative to bracket 0; a row with a negative first sign takes the conjugate
        prod = {}
        for pat in pl["patterns"]:
            b = bj[0].copy()
            for j in range(1, nsp):
                b *= np.conj(bj[j]) if (pat >> (j - 1)) & 1 else bj[j]
            prod[pat] = b
    total = np.zeros(B)
    dT = np.ascontiguousarray(dinv.T)                       # (T, B)
    if nsp:
        Yre = np.concatenate([dT * prod[k].real.T for k in pl["patterns"]], 0)
        Yim = np.concatenate([dT * prod[k].imag.T for k in pl["patterns"]], 0)
    for h in range(pl["H"]):
        sp = pl["sparse"][h]
        F = (spow[:, sp["rk"]] * Q[:, sp["rq"]]).T            # (rows, B): s^(-k/2) times the monomial
        if nsp:
            # U = sum X Re(br), V = sum X Im(br); |M|^2 parity even through (|U + iV|^2 + |U - iV|^2) / 2
            U = np.add.reduceat((sp["S_re"] @ Yre) * F, sp["gstart"], axis=0)
            V = np.add.reduceat((sp["S_im"] @ Yim) * F, sp["gstart"], axis=0)
            total += (U.real ** 2 + U.imag ** 2 + V.real ** 2 + V.imag ** 2).sum(0)
        else:
            A = np.add.reduceat((sp["S_re"] @ dT) * F, sp["gstart"], axis=0)
            total += (A.real ** 2 + A.imag ** 2).sum(0)
    if not coherent:
        total /= nP
    if st.get("coupling_mode", "global") == "global":
        total *= st["coupling"] * a_s ** st["k_s"]
    return total


# =============================================================================
# Revision 2 (versions 3-6; the r2 names and v6): every SM-legal diagram, one coherent sum
# =============================================================================
_ENUM_MAX = 50000       # above this many legal trees, orbits come from uniform tree sampling (up to this many labelled
                        # diagrams) instead of a listing; no recipe name reaches it (module docstring, revision 2)
_V6_MONOS = 1           # v6: bracket monomials per (diagram, configuration) (module docstring, version 6)


def _legal_tables(out, N, protected, mass, hgg):
    """Counting and listing of the SM-legal cubic trees of the all-outgoing particles `out` (version 3's vertices,
    ggH only on an external H, a massless or light line only on a protected channel: version 3's rules), with leg
    N-1 as the root. Returns (count, gen, sample): gen() lists every tree, sample(rng) draws one uniformly."""
    tab = _line_table(hgg)
    top = frozenset(range(N - 1))
    x_top = _anti(out[N - 1])

    @functools.lru_cache(maxsize=None)
    def splits(S):
        legs = sorted(S)
        res = []
        for r in range(len(legs) - 1):
            for comb in itertools.combinations(legs[1:], r):
                A = frozenset((legs[0],) + comb)
                res.append((A, S - A))
        return res

    def opts(S, A, Xa, B, Xb):
        is_top = S == top
        res = []
        for X, cls in tab.get((Xa, Xb), ()):
            if is_top and X != x_top:
                continue
            if cls == "HGG" and not ((Xa == 25 and len(A) == 1) or (Xb == 25 and len(B) == 1)
                                     or (is_top and out[N - 1] == 25)):
                continue
            if not is_top and mass(X) <= _LIGHT and not protected(_canonical(S, N)):
                continue
            res.append((X, cls))
        return res

    @functools.lru_cache(maxsize=None)
    def cnt(S):
        if len(S) == 1:
            return {out[next(iter(S))]: 1}
        res = {}
        for A, B in splits(S):
            ca, cb = cnt(A), cnt(B)
            for Xa, na in ca.items():
                for Xb, nb in cb.items():
                    for X, _c in opts(S, A, Xa, B, Xb):
                        res[X] = res.get(X, 0) + na * nb
        return res

    @functools.lru_cache(maxsize=None)
    def ways(S, X):
        """[(A, B, Xa, Xb, cls, weight)] building cluster S with outgoing line X."""
        res = []
        for A, B in splits(S):
            for Xa, na in cnt(A).items():
                for Xb, nb in cnt(B).items():
                    for Xp, cls in opts(S, A, Xa, B, Xb):
                        if Xp == X:
                            res.append((A, B, Xa, Xb, cls, na * nb))
        return res

    def gen(S=top, X=x_top):
        if len(S) == 1:
            yield next(iter(S))
            return
        for A, B, Xa, Xb, cls, _w in ways(S, X):
            for ta in list(gen(A, Xa)):
                for tb in gen(B, Xb):
                    yield (X, cls, S, ta, tb)

    def sample(rng, S=top, X=x_top):
        if len(S) == 1:
            return next(iter(S))
        w = ways(S, X)
        p = np.array([x[5] for x in w], float)
        A, B, Xa, Xb, cls, _w = w[int(rng.choice(len(w), p=p / p.sum()))]
        return (X, cls, S, sample(rng, A, Xa), sample(rng, B, Xb))

    return cnt(top).get(x_top, 0), gen, sample


def _tree_to_diagram(tree, N):
    """A nested legal tree -> the diagram dict of _tree_legal (merges, root, props with legs/pdg/x, vertex_class)."""
    merges, props, vcls = [], [], []

    def build(node):
        if not isinstance(node, tuple):
            return int(node)
        X, cls, S, a, b = node
        ia, ib = build(a), build(b)
        merges.append([ia, ib])
        vcls.append(cls)
        Sc = _canonical(S, N)
        props.append({"legs": list(Sc), "pdg": int(X if tuple(sorted(S)) == Sc else _anti(X)), "x": int(X)})
        return N + len(merges) - 1
    _X, cls, _S, a, b = tree
    ia, ib = build(a), build(b)
    vcls.append(cls)
    return {"merges": merges, "root": [ia, ib, N - 1], "props": props, "vertex_class": vcls}


def _dkey(props):
    return frozenset((tuple(p["legs"]), int(p["pdg"])) for p in props)


def _img_key(key, perm, N):
    """The key of a diagram relabelled by `perm` (slot i -> perm[i]); a line seen from its other side is the
    antiparticle (only when identical beams are exchanged)."""
    res = []
    for S, pdg in key:
        S2 = tuple(sorted(perm[i] for i in S))
        Sc = _canonical(S2, N)
        res.append((Sc, pdg if S2 == Sc else _anti(pdg)))
    return frozenset(res)


def _legal_orbits(rng, out, N, protected, mass, hgg, perms, cap):
    """Every legal diagram of the process, grouped into orbits of the identical-leg permutations: returns
    (representatives [(diagram, stabiliser perms, orbit size)], legal count, labelled diagrams kept, subsampled).
    Called with cap = _ENUM_MAX: a listed set (count <= _ENUM_MAX) is kept whole; above it, sampled whole orbits up to
    `cap` labelled diagrams."""
    count, gen, sample = _legal_tables(out, N, protected, mass, hgg)
    if count == 0:
        return [], 0, 0, False
    reps, seen = [], set()

    def add(tree):
        d = _tree_to_diagram(tree, N)
        key = _dkey(d["props"])
        if key in seen:
            return None
        imgs = [_img_key(key, p, N) for p in perms]
        stab = [p for p, k in zip(perms, imgs) if k == key]
        orbit = set(imgs)
        seen.update(orbit)
        return (d, stab, len(orbit))

    if count <= _ENUM_MAX:
        for tree in gen():
            r = add(tree)
            if r is not None:
                reps.append(r)
        if sum(r[2] for r in reps) != count:
            raise RuntimeError("legal diagram set not closed under the identical-leg permutations")
        if count <= cap:
            return reps, count, count, False
        keep, k = [], 0
        for i in rng.permutation(len(reps)):
            if k + reps[i][2] <= cap:
                keep.append(int(i))
                k += reps[i][2]
        return [reps[i] for i in sorted(keep)], count, k, True
    k, misses = 0, 0
    for _ in range(50 * cap):          # too many to list: uniform trees (orbits in proportion to their size), whole
        r = add(sample(rng))           # orbits while they fit; stops after 500 draws in a row add nothing
        if r is None or k + r[2] > cap:
            misses += 1
            if misses >= 500:
                break
            continue
        misses = 0
        reps.append(r)
        k += r[2]
        if k >= 0.95 * cap:
            break
    return reps, count, k, True


def _line_groups(d, stab, N):
    """Group the lines of a diagram that its stabiliser maps onto each other (they must share a power)."""
    chan = [tuple(p["legs"]) for p in d["props"]]
    where = {c: j for j, c in enumerate(chan)}
    parent = list(range(len(chan)))

    def root(j):
        while parent[j] != j:
            j = parent[j]
        return j
    for perm in stab:
        for j, c in enumerate(chan):
            parent[root(j)] = root(where[_canonical([perm[i] for i in c], N)])
    return [root(j) for j in range(len(chan))]


def _draw_process_r2(rng, name, ver, ini, fin, target):
    """Revision 2, versions 3-6: every legal diagram (no cap; orbits sampled only above _ENUM_MAX legal trees), one
    numerator per permutation
    orbit, summed coherently over the orbit's images instead of an incoherent average over the permutations."""
    pdgs = ini + fin
    n, N = len(fin), len(pdgs)
    m_legs = [leg_mass(p) for p in pdgs]
    cut = [i >= 2 and m_legs[i] == 0.0 and abs(pdgs[i]) not in _NEUTRINOS for i in range(N)]
    guard = [cut[i] or m_legs[i] > 0.0 for i in range(N)]

    def protected(S):                                       # version 1's rule
        if 1 not in S:
            return sum(cut[i] for i in S) >= 2 or sum(m_legs[i] for i in S) > _LIGHT
        G = [i for i in S if i >= 2]
        F = [i for i in range(2, N) if i not in S]
        return any(guard[i] for i in G) and any(guard[i] for i in F)

    m_scalar = float(np.exp(rng.uniform(*np.log(_SCALAR_MASS)))) if rng.random() >= 0.5 else 125.0
    if 25 in fin:
        m_scalar = 125.0
    mass = lambda X: m_scalar if abs(X) == 25 else leg_mass(X)
    ratio = {}

    def width(pdg):
        a = abs(pdg)
        if a not in ratio:
            ratio[a] = float(np.exp(rng.uniform(*np.log(_WIDTH_RATIO))))
        return ratio[a]

    perms = _identical_perms(pdgs)
    out = [_anti(p) for p in ini] + list(fin)
    reps, n_legal, k_all, sub = _legal_orbits(rng, out, N, protected, mass, ver >= 5, perms, _ENUM_MAX)
    if not reps:
        return None
    diagrams = []
    for d, stab, osize in reps:
        groups, gpow, spinors = _line_groups(d, stab, N), {}, []
        for j, pr in enumerate(d["props"]):
            S, M = pr["legs"], mass(pr["pdg"])
            s_type = 1 not in S
            pr["mass"] = M
            pr["width"] = width(pr["pdg"]) * M if (s_type and M > 0) else 0.0
            if ver >= 4 and M == 0.0 and s_type and len(S) == 2 and all(m_legs[i] == 0.0 for i in S):
                pr["power"] = 1.0                           # v4: the numerator vanishes instead
                spinors.append({"legs": list(S), "soft": [i for i in S if pdgs[i] in _GAUGE]})
            elif M > _LIGHT:
                pr["power"] = 1.0
            else:
                if groups[j] not in gpow:
                    gpow[groups[j]] = float(rng.choice([0.5, 1.0]))
                pr["power"] = gpow[groups[j]]
        if ver >= 4:
            d["spinors"] = spinors
        d["w"], d["orbit"] = 1.0 / len(stab), int(osize)
        diagrams.append(d)

    e = np.sqrt(4 * np.pi * ALPHA_EW)
    for d in diagrams:                                      # couplings from the vertices, as versions 3-5
        nq, ne, nh = (sum(c == k for c in d["vertex_class"]) for k in ("QCD", "QED", "HGG"))
        d["as_pow"] = 0.5 * nq + nh
        d["cpl"] = float((4 * np.pi) ** (0.5 * nq) * e ** ne * (e / (3 * np.pi)) ** nh)
        d["n_ew"] = ne + nh
    k_s = int(round(2 * max(d["as_pow"] for d in diagrams)))
    order = [0, 0, k_s, int(max(d["n_ew"] for d in diagrams))]

    pairs = list(itertools.combinations(range(N), 2))
    st = {
        "name": name, "version": ver, "revision": 2, "target_shape": bool(target), "pdg_ids": pdgs,
        "m_legs": m_legs, "nfinal": n, "sqrts": [float(np.ceil(max(25.0, 1.05 * sum(m_legs[2:])))), SQRTS_MAX],
        "k_s": k_s, "coupling": 1.0, "coupling_mode": "vertex", "coherent": True, "order": order,
        "n_legal": int(n_legal), "k_all": int(k_all), "subsampled": bool(sub),
        "internal_masses": {25: m_scalar}, "diagrams": diagrams, "pairs": [list(p) for p in pairs], "perms": perms,
    }
    if ver < 6:
        H = int(rng.integers(_H[0], _H[1] + 1))
        numer = []
        for _h in range(H):                                 # versions 3-5's numerators, one per orbit
            row = []
            for d in diagrams:
                sp = d.get("spinors", [])
                banned = {tuple(s["legs"]) for s in sp}
                allowed = [i for i, p in enumerate(pairs) if p not in banned]
                deg = int(rng.integers(0, 3))
                monos = [{"a": float(rng.normal()),
                          "pairs": sorted(int(allowed[x]) for x in rng.choice(len(allowed), size=deg))}
                         for _ in range(int(rng.integers(1, 4)))]
                e_sum = sum(p["power"] for p in d["props"])
                term = {"c": [float(rng.normal()), float(rng.normal())], "deg": deg, "monos": monos,
                        "k": 2 * deg + len(sp) - 2 * e_sum - 4 + N}
                if sp:
                    term["sig"] = [int(x) for x in rng.choice([-1, 1], size=len(sp))]
                row.append(term)
            numer.append(row)
        st["numerators"] = numer
    else:
        _numerators_v6(rng, st, out)
        if not st["configs"]:
            return None
    st["sha"] = hashlib.sha256(json.dumps(st, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return st


# =============================================================================
# Version 6: spinor-bracket numerators, per-leg helicities, tree-level selection rules
# =============================================================================
_LEPTON_NU = frozenset((12, 14, 16))
_ALL_FERMIONS = frozenset(range(1, 7)) | frozenset(range(11, 17))


def _hel2_options(p):
    """2h per leg (all-outgoing): fermions +-1, g/gamma +-2, Z/W -2, 0, 2 (massive vector), H 0."""
    a = abs(p)
    if a in (21, 22):
        return (-2, 2)
    if a in (23, 24):
        return (-2, 0, 2)
    if a == 25:
        return (0,)
    return (-1, 1)


def _v6_configs(out, m_legs, has_scalar):
    """Every helicity configuration (2h per leg) that survives the configuration-level rule: with every external
    leg massless and no scalar anywhere, fewer than two negative or fewer than two positive helicities vanish."""
    allm = all(m == 0.0 for m in m_legs) and not has_scalar
    res = []
    for lam in itertools.product(*[_hel2_options(p) for p in out]):
        if allm and (sum(h < 0 for h in lam) < 2 or sum(h > 0 for h in lam) < 2):
            continue
        res.append(lam)
    return res


def fermion_lines(d, out, N):
    """The fermion lines of diagram d: [(leg i, leg j, chiral)], i, j its external ends (all-outgoing), chiral when
    the line passes a W vertex or ends on a neutrino (the SM's W couples to left-handed fermions only, and only
    left-handed neutrinos exist)."""
    X = {i: out[i] for i in range(N)}
    for c, pr in zip(range(N, N + len(d["merges"])), d["props"]):
        X[c] = pr["x"]
    verts = [list(m) + [N + v] for v, m in enumerate(d["merges"])] + [list(d["root"])]
    prod = {N + v: v for v in range(len(d["merges"]))}
    cons = {}
    for v, m in enumerate(d["merges"]):
        for e in m:
            cons[e] = v
    for e in d["root"]:
        cons[e] = len(verts) - 1
    isf = lambda e: abs(X[e]) in _ALL_FERMIONS
    lines, done = [], set()
    for i in range(N):
        if not isf(i) or i in done:
            continue
        e, v, chiral = i, cons[i], abs(out[i]) in _LEPTON_NU
        while True:
            chiral = chiral or any(abs(X[x]) == 24 for x in verts[v])
            e = [x for x in verts[v] if x != e and isf(x)][0]
            if e < N:
                break
            v = prod[e] if cons[e] == v else cons[e]
        chiral = chiral or abs(out[e]) in _LEPTON_NU
        done.update((i, e))
        lines.append((i, e, chiral))
    return lines


def _alive_mask(lines, m_legs, out, MU):
    """Which configurations MU (C, N) of 2h a diagram with these fermion lines allows: helicity conserved along a
    massless line (its ends opposite in the all-outgoing convention), left-handed on a chiral one."""
    ok = np.ones(len(MU), bool)
    for i, j, chiral in lines:
        if m_legs[i] != 0.0 or m_legs[j] != 0.0:
            continue
        ok &= MU[:, i] == -MU[:, j]
        if chiral:
            f = i if out[i] > 0 else j                      # the fermion (not the antifermion) end
            ok &= MU[:, f] == -1
    return ok


def _bracket_skeleton(rng, hel2, gauge, forced, N):
    """One monomial's stubs before pairing (version 6): per leg the signed angle (A) and square (S) degrees with
    A - S = -2h (little-group weight t^(-2h)); gauge legs (massless final g, gamma) A + S = 0 (the monomial does
    not vanish as the leg goes soft) and they alone take denominators, each paired with a beam, which gets a stub
    back; first the forced bracket of each massless two-leg line. Returns (brackets so far, A, S)."""
    A, S = [0] * N, [0] * N
    for l in range(N):
        if gauge[l]:
            A[l], S[l] = -hel2[l] // 2, hel2[l] // 2
        else:
            A[l], S[l] = max(-hel2[l], 0), max(hel2[l], 0)
    brs = []
    for i, k in forced:
        gl = [x for x in (i, k) if gauge[x]]
        if gl:
            g = gl[int(rng.integers(len(gl)))]
            typ = 1 if hel2[g] > 0 else 0                   # the gauge leg's numerator stub: [ ] for h = +1
        else:
            typ = int(rng.integers(2))
        brs.append((typ, i, k, 1))
        T = S if typ else A
        T[i] -= 1
        T[k] -= 1
    for l in range(N):
        while not gauge[l] and (A[l] < 0 or S[l] < 0):
            A[l] += 1
            S[l] += 1
    for typ, T in ((0, A), (1, S)):
        for l in range(N):
            while gauge[l] and T[l] < 0:                    # <l r>^-1 or [l r]^-1, r a beam
                r = _lean(rng, T, (0, 1))                   # the beam with fewer stubs (a matching must exist)
                brs.append((typ, l, r, -1))
                T[l] += 1
                T[r] += 1
    return brs, A, S


def _lean(rng, count, legs):
    """Among `legs`, one with the fewest stubs (random among ties): keeps every leg below half of a bracket type's
    stubs, the condition for the stubs to pair up without a self-bracket <ll>."""
    c = np.array([count[l] for l in legs])
    cand = [l for l, x in zip(legs, c) if x == c.min()]
    return cand[int(rng.integers(len(cand)))]


def _lean2(rng, A, S, legs, forbid=()):
    """The leg for a dipole (one stub of each type), chosen so that a matching stays possible: after adding it, the
    largest excess of a leg, or of a forbidden pair (whose two legs cannot pair with each other), over all the other
    stubs of a type is smallest (random among ties)."""
    groups = [(l,) for l in range(len(A))] + [tuple(f) for f in forbid]
    best, cand = None, []
    for l in legs:
        A[l] += 1
        S[l] += 1
        sa, ss = sum(A), sum(S)
        x = max(max(2 * sum(A[i] for i in g) - sa, 2 * sum(S[i] for i in g) - ss) for g in groups)
        A[l] -= 1
        S[l] -= 1
        if best is None or x < best:
            best, cand = x, [l]
        elif x == best:
            cand.append(l)
    return cand[int(rng.integers(len(cand)))]


def _pair_stubs(rng, stubs, forbid):
    """A random perfect matching of the stubs (leg labels) with no <ll> and no forbidden pair, None if none was
    found: the leg with the most stubs left takes a partner drawn in proportion to the others' stubs."""
    if not stubs:
        return []
    for _ in range(20):                                     # plain shuffles first (cheap), then the greedy draw
        p = [stubs[i] for i in rng.permutation(len(stubs))]
        prs = [(p[2 * m], p[2 * m + 1]) for m in range(len(p) // 2)]
        if all(a != b and frozenset((a, b)) not in forbid for a, b in prs):
            return prs
    for _ in range(50):
        left = {}
        for l in stubs:
            left[l] = left.get(l, 0) + 1
        prs = []
        while left:
            top = max(left.values())
            a = sorted(l for l, c in left.items() if c == top)
            a = a[int(rng.integers(len(a)))]
            cand = [l for l in sorted(left) if l != a and frozenset((a, l)) not in forbid]
            if not cand:
                break
            w = np.array([left[l] for l in cand], float)
            b = cand[int(rng.choice(len(cand), p=w / w.sum()))]
            prs.append((a, b))
            for x in (a, b):
                left[x] -= 1
                if not left[x]:
                    del left[x]
        if not left:
            return prs
    return None


def _finish_monomial(rng, brs, A, S, gauge, forbid, n_target):
    """Pad with dipoles (one angle and one square stub on a non-gauge leg: weight 0, one more bracket) to n_target
    brackets (net count), then pair the stubs. Returns the bracket list, or None if no pairing was found."""
    A, S = list(A), list(S)
    nong = [l for l in range(len(A)) if not gauge[l]]
    n_now = sum(x[3] for x in brs) + (sum(A) + sum(S)) // 2
    if (sum(A) + sum(S)) % 2 or n_target < n_now or (n_target - n_now) % 2:
        raise RuntimeError("inconsistent bracket count")
    for _ in range(n_target - n_now):
        l = _lean2(rng, A, S, nong, forbid)
        A[l] += 1
        S[l] += 1
    res = list(brs)
    for typ, T in ((0, A), (1, S)):
        prs = _pair_stubs(rng, [l for l in range(len(T)) for _ in range(T[l])], forbid)
        if prs is None:
            return None
        res += [(typ, int(a), int(b), 1) for a, b in prs]
    return res


def _numerator_v6(rng, hel2, gauge, forced, N, n_mono=_V6_MONOS):
    """n_mono bracket monomials of one (diagram, configuration), all with the same weights and the same net bracket
    count: the largest minimal count of their skeletons plus 2 delta, delta ~ U{0,1,2} (version 1's degree range).
    Returns (monomials, bracket count)."""
    nong = [l for l in range(N) if not gauge[l]]
    skels = []
    for _ in range(n_mono):
        brs, A, S = _bracket_skeleton(rng, hel2, gauge, forced, N)
        if sum(A) % 2:                                      # odd stub count: one dipole on a non-gauge leg
            l = _lean2(rng, A, S, nong, [frozenset(p) for p in forced])
            A[l] += 1
            S[l] += 1
        skels.append((brs, A, S, sum(x[3] for x in brs) + (sum(A) + sum(S)) // 2))
    forbid = {frozenset(p) for p in forced}
    target = max(s[3] for s in skels) + 2 * int(rng.integers(0, 3))
    for _ in range(20):
        monos = [_finish_monomial(rng, b, A, S, gauge, forbid, target) for b, A, S, _n in skels]
        if all(m is not None for m in monos):
            return [{"a": float(rng.normal()), "br": [list(x) for x in m]} for m in monos], target
        target += 2
    raise RuntimeError("no bracket monomial found")


def _neg(mu):
    return tuple(-h for h in mu)


def _canon(mu):
    """A configuration is canonical when its first non-zero 2h is negative (one of each parity pair)."""
    for h in mu:
        if h:
            return h < 0
    return True


def _v6_lambda(D, out, m_legs, perms):
    """(configurations after the configuration-level rule, alive[r, c]: representative r allows configuration c,
    the surviving configurations: those some image of some representative allows)."""
    N = len(out)
    has_scalar = any(abs(p["pdg"]) == 25 for d in D for p in d["props"]) or 25 in [abs(p) for p in out]
    conf = _v6_configs(out, m_legs, has_scalar)
    if not conf:
        return conf, None, []
    C = np.array(conf, np.int64)
    alive = np.stack([_alive_mask(fermion_lines(d, out, N), m_legs, out, C) for d in D])
    code = {c: i for i, c in enumerate(conf)}
    keep = np.zeros(len(C), bool)
    for perm in perms:
        idx = np.array([code[tuple(int(x) for x in row)] for row in C[:, perm]])
        keep |= alive[:, idx].any(0)
    return conf, alive, [conf[i] for i in np.flatnonzero(keep)]


def _numerators_v6(rng, st, out):
    """Helicity configurations, selection rules and bracket numerators of a version-6 structure (module docstring).
    One numerator per (orbit representative, configuration) that the representative allows; a configuration whose
    parity partner is allowed too takes the partner's numerator with every bracket conjugated (<ij> <-> [ij])."""
    pdgs, m_legs, N, D = st["pdg_ids"], st["m_legs"], len(st["pdg_ids"]), st["diagrams"]
    conf, alive, lam = _v6_lambda(D, out, m_legs, st["perms"])
    st["configs"] = []
    if not lam:
        return
    gauge = [i >= 2 and pdgs[i] in _GAUGE and m_legs[i] == 0.0 for i in range(N)]
    for d in D:
        d["flines"] = [list(x) for x in fermion_lines(d, out, N)]
    code = {c: i for i, c in enumerate(conf)}
    lset = set(lam)
    numer = []
    for r, d in enumerate(D):
        forced = [tuple(s["legs"]) for s in d.get("spinors", [])]
        e_sum = sum(p["power"] for p in d["props"])
        for c in np.flatnonzero(alive[r]):
            mu = conf[c]
            if mu not in lset:
                continue
            nm = _neg(mu)
            if not _canon(mu) and nm in code and alive[r, code[nm]] and nm in lset:
                continue                                    # the partner's numerator, conjugated
            monos, nb = _numerator_v6(rng, mu, gauge, forced, N)
            numer.append({"d": r, "hel": list(mu), "c": [float(rng.normal()), float(rng.normal())],
                          # mass dimension 4 - N: n_b - k - 2 E = 4 - N
                          "k": nb - 2 * e_sum + N - 4, "monos": monos})
    st["configs"] = [list(x) for x in lam]
    st["numerators"] = numer


def _spinor_brackets(P, m_legs, lg=None):
    """Bracket table (B, 2 N^2): column (typ*N + i)*N + j holds <ij> (typ 0) or [ij] (typ 1) (1 on the unused
    diagonal), and s = (p_a + p_b)^2 (B,). Spinors of the physical (positive-energy) momenta, [ij] = conj(<ij>), |<ij>|^2 = 2 p_i.p_j.
    A massive leg enters through p_flat, its light-cone component along its CM-frame direction ((E + |k|)/2 (1, k/|k|)
    in the CM frame, P = p_a + p_b), so its helicity is the usual one (quantised along the momentum in the CM
    frame), the same for every term and symmetric in the beams. lg (N,) scales lambda_i -> t lambda_i, lambda~_i -> lambda~_i / t (the little-group test)."""
    P = np.asarray(P, np.float64)
    B, N = P.shape[:2]
    m2 = np.array([m * m for m in m_legs], float)
    Pt = P[:, 0] + P[:, 1]
    s_tot = _mink2(Pt)
    pf = P.copy()
    u = Pt / np.sqrt(s_tot)[:, None]
    for l in np.flatnonzero(m2 > 0):
        # in the CM frame p_flat = (E + |k|)/2 (1, k/|k|), the light-cone component along the momentum (p_flat -> p as
        # m -> 0): p_flat = (E + |k|)/(2|k|) p - m^2/(2|k|) u, u = P/sqrt(s), E = p.u, |k| = sqrt(E^2 - m^2)
        E = _dot(P[:, l], u)
        k = np.sqrt(np.maximum(E * E - m2[l], 1e-300))
        pf[:, l] = ((E + k) / (2 * k))[:, None] * P[:, l] - (m2[l] / (2 * k))[:, None] * u
    E, x, y, z = pf[..., 0], pf[..., 1], pf[..., 2], pf[..., 3]
    up = z >= 0
    sp = np.sqrt(np.maximum(E + z, 0.0))
    sm = np.sqrt(np.maximum(E - z, 0.0))
    lam0 = np.where(up, sp, (x - 1j * y) / np.where(up, 1.0, sm))
    lam1 = np.where(up, (x + 1j * y) / np.where(up, sp, 1.0), sm)
    lt0, lt1 = np.conj(lam0), np.conj(lam1)
    if lg is not None:
        t = np.asarray(lg, np.complex128)[None, :]
        lam0, lam1, lt0, lt1 = lam0 * t, lam1 * t, lt0 / t, lt1 / t
    ang = lam0[:, :, None] * lam1[:, None, :] - lam1[:, :, None] * lam0[:, None, :]
    sq = lt0[:, :, None] * lt1[:, None, :] - lt1[:, :, None] * lt0[:, None, :]
    eye = np.eye(N, dtype=bool)[None]
    res = np.concatenate([np.where(eye, 1.0, ang).reshape(B, N * N), np.where(eye, 1.0, sq).reshape(B, N * N)], 1)
    return res, s_tot


_PLANS6 = {}


def _plan6(st):
    """Per helicity unit (one configuration, or a parity pair whose every term pairs by conjugation) its alive terms:
    image index into the propagator table, coefficient, the bracket factors as gathers into _spinor_brackets."""
    pl = _PLANS6.get(st["sha"])
    if pl is not None:
        return pl
    base = _plan(st)
    N, D, K = len(st["pdg_ids"]), st["diagrams"], len(st["diagrams"])
    perms = np.array(st["perms"], np.int64)
    out = [_anti(p) if i < 2 else p for i, p in enumerate(st["pdg_ids"])]
    lam = [tuple(c) for c in st["configs"]]
    lset = set(lam)
    keys = [(e["d"], tuple(e["hel"])) for e in st["numerators"]]
    eid = {k: i for i, k in enumerate(keys)}
    nE = len(keys)
    Mx = max(len(e["monos"]) for e in st["numerators"])
    Fx = max(len(m["br"]) for e in st["numerators"] for m in e["monos"])
    typ = np.zeros((nE, Mx, Fx), np.int64)
    bi = np.zeros((nE, Mx, Fx), np.int64)
    bj = np.zeros((nE, Mx, Fx), np.int64)
    be = np.zeros((nE, Mx, Fx))
    pad = np.ones((nE, Mx, Fx), bool)
    am = np.zeros((nE, Mx))
    ce = np.zeros(nE, np.complex128)
    ke = np.zeros(nE)
    for n_, e in enumerate(st["numerators"]):
        ce[n_], ke[n_] = complex(*e["c"]), e["k"]
        for m, mo in enumerate(e["monos"]):
            am[n_, m] = mo["a"]
            for f, (t_, i_, j_, x_) in enumerate(mo["br"]):
                typ[n_, m, f], bi[n_, m, f], bj[n_, m, f], be[n_, m, f], pad[n_, m, f] = t_, i_, j_, x_, False
    # alive[r][code of mu]
    conf_all = _v6_configs(out, st["m_legs"], True)        # every configuration (rule applied per lam below)
    code = {c: i for i, c in enumerate(conf_all)}
    CA = np.array(conf_all, np.int64)
    alive = np.stack([_alive_mask(d["flines"], st["m_legs"], out, CA) for d in D])
    units, done = [], set()
    for lm in lam:
        if lm in done:
            continue
        nl = _neg(lm)
        MU = np.array(lm, np.int64)[perms]                  # (nP, N): slot i of image s carries lam[perm_s[i]]
        cm = np.array([code[tuple(int(x) for x in row)] for row in MU])
        cn = np.array([code[_neg(tuple(int(x) for x in row))] for row in MU])
        a1, a2 = alive[:, cm], alive[:, cn]                 # (K, nP)
        rows = []
        for s in range(len(perms)):
            mu = conf_all[cm[s]]
            for r in np.flatnonzero(a1[:, s]):
                r = int(r)
                if (r, mu) in eid:
                    rows.append((s * K + r, eid[(r, mu)], False, s))
                else:
                    rows.append((s * K + r, eid[(r, _neg(mu))], True, s))
        if nl != lm and nl in lset and np.array_equal(a1, a2):
            units.append((lm, 2.0, rows))                   # its partner -lam is U - iV
            done.update((lm, nl))
        else:
            units.append((lm, 1.0, rows))
            done.add(lm)
    U = []
    C = 2 * N * N                                           # bracket columns of _spinor_brackets
    for lm, w, rows in units:
        t = np.array([x[0] for x in rows], np.int64)
        e = np.array([x[1] for x in rows], np.int64)
        cj = np.array([x[2] for x in rows], np.int64)
        sg = perms[np.array([x[3] for x in rows], np.int64)]
        ty = typ[e] ^ cj[:, None, None]
        flat = lambda b: np.take_along_axis(sg, b.reshape(len(e), -1), 1).reshape(b.shape)
        fi = (ty * N + flat(bi[e])) * N + flat(bj[e])
        pd = pad[e]
        present = ~pd.all(-1)                               # (terms, M): the monomials that exist
        tt, mm = np.nonzero(present)                        # one row per monomial, grouped by term
        # a factor <ij>^e is row col of the stacked table [brackets; 1/brackets] (col + C when e = -1); rows are
        # bucketed by their factor count
        ci = np.where(be[e][tt, mm] < 0, fi[tt, mm] + C, fi[tt, mm])
        ok = ~pd[tt, mm]
        nf = ok.sum(1)
        buckets = []
        for f in np.unique(nf):
            rws = np.flatnonzero(nf == f)
            buckets.append((rws, np.stack([ci[r][ok[r]] for r in rws]).astype(np.int32)))
        starts = np.concatenate([[0], np.cumsum(present.sum(1))[:-1]]).astype(np.int64)
        ku, kidx = np.unique(ke[e][tt], return_inverse=True)
        U.append({"lam": lm, "w": w, "t": t, "c": ce[e], "a": am[e][tt, mm], "ku": ku, "kidx": kidx,
                  "buckets": buckets, "nrow": len(tt), "starts": starts})
    off = 0
    kall = np.unique(np.concatenate([u["ku"] for u in U]))  # the s-powers, factors 2C.. of the product table
    for u in U:
        u["off"] = off
        off += u["nrow"]
        u["kfac"] = 2 * C + np.searchsorted(kall, u["ku"])[u["kidx"]]
        # one coefficient per monomial: c of its term times its a (one monomial per term unless _V6_MONOS > 1)
        u["ca"] = (u["c"][np.repeat(np.arange(len(u["t"])), np.diff(np.append(u["starts"], u["nrow"])))]
                   * u["a"])
    pl = {"base": base, "units": U, "C": C, "kall": kall, "nrows": int(off),
          "nterms": int(sum(len(u["t"]) for u in U))}
    for u in U:                     # one prefix tree per unit: its products stay in cache for the unit's sums
        u["trie"] = _bracket_trie([dict(u, off=0)], 2 * C + len(kall), u["nrow"])
    _PLANS6[st["sha"]] = pl
    return pl


def _bracket_trie(U, n_fac, R):
    """The monomials of every unit as one prefix tree: each monomial is a product of factors (rows of the stacked
    bracket table), ordered by how many monomials use the factor (most first, then by index), so that a product
    shared by many monomials (the same brackets on the beams and the boson in every permutation image, the same
    leading factors across configurations) is computed once. Built level by level with np.unique; a level's node is
    its parent's product times one factor. Returns per level (parent node, factor) and the monomials ending there."""
    L = np.zeros(R, np.int64)
    F = max((idx.shape[1] for u in U for _r, idx in u["buckets"]), default=1)
    F += 1                                                  # the s-power s^(-k/2) is one more factor
    fac = np.full((R, F), n_fac, np.int64)
    for u in U:
        for rws, idx in u["buckets"]:
            fac[u["off"] + rws, :idx.shape[1]] = idx
            fac[u["off"] + rws, idx.shape[1]] = u["kfac"][rws]
            L[u["off"] + rws] = idx.shape[1] + 1
    freq = np.bincount(fac[fac < n_fac], minlength=n_fac + 1)
    freq[n_fac] = -1
    key = (freq.max() - freq[fac]) * (n_fac + 1) + fac     # most used first, then by index; padding last
    fac = np.take_along_axis(fac, np.argsort(key, axis=1, kind="stable"), 1)
    node = np.zeros(R, np.int64)
    levels = []
    for lev in range(F):
        act = np.flatnonzero(L > lev)
        k = node[act] * (n_fac + 1) + fac[act, lev]
        uq, inv = np.unique(k, return_inverse=True)
        node[act] = inv
        end = L[act] == lev + 1
        levels.append({"parent": uq // (n_fac + 1), "fac": uq % (n_fac + 1), "end_rows": act[end],
                       "end_nodes": inv[end]})
    return levels


def _trie_products(trie, BT, R):
    """Every monomial's bracket product (R, B), one multiplication per prefix-tree node."""
    G = np.empty((R, BT.shape[1]), np.complex128)
    vals = None
    for lev in trie:
        f = np.take(BT, lev["fac"], axis=0)
        vals = f if vals is None else np.multiply(np.take(vals, lev["parent"], axis=0), f, out=f)
        if len(lev["end_rows"]):
            G[lev["end_rows"]] = vals[lev["end_nodes"]]
    return G


def _dinv(pl, P, scale):
    """Propagators times couplings of every image (B, T), as in _eval_chunk."""
    N = P.shape[1]
    sign = np.array([-1.0, -1.0] + [1.0] * (N - 2))
    q = P * sign[None, :, None]
    s = _mink2(np.einsum("cn,bnk->bck", pl["chmask"], q))[:, pl["ptC"]]
    M, G = pl["ptM"] * scale, pl["ptW"] * scale
    val = s - M * M + 1j * M * G
    half = pl["ptE"] != 1.0
    if half.any():
        val[:, half] = np.where(M[half] > 0, np.sqrt(val[:, half]), np.sqrt(np.abs(s[:, half])))
    inv = 1.0 / val
    dinv = inv[:, pl["den"][:, 0]]
    for j in range(1, pl["den"].shape[1]):
        dinv *= inv[:, pl["den"][:, j]]
    a_s = _alpha_s(P, scale)
    return dinv * (pl["const"][None, :] * a_s[:, None] ** pl["as_pow"][None, :])


def _unit_UV(u, BT, dinvT):
    """U = sum X Re(poly), V = sum X Im(poly) of one helicity unit (A = U + iV; its parity image U - iV), from its
    monomial products (its prefix tree over the factor table BT, _trie_products; s-power included) and the
    images' propagator table dinvT (T, B); X = propagators x c, poly = sum of a x monomial. Four real reductions,
    no temporaries beyond X."""
    g = _trie_products(u["trie"], BT, u["nrow"])
    X = np.take(dinvT, u["t"], axis=0)
    if u["nrow"] == len(u["t"]):
        X *= u["ca"][:, None]
    else:                                                   # several monomials per term: sum them first
        X *= u["c"][:, None]
        g = np.add.reduceat(g * u["a"][:, None], u["starts"], axis=0)
    gr, gi, xr, xi = g.real, g.imag, X.real, X.imag
    U = np.einsum("rb,rb->b", xr, gr) + 1j * np.einsum("rb,rb->b", xi, gr)
    V = np.einsum("rb,rb->b", xr, gi) + 1j * np.einsum("rb,rb->b", xi, gi)
    return U, V


def _brtable(br, s_tot=None, kall=()):
    """[brackets; 1/brackets; s^(-k/2) for each k of kall] transposed (2C + len(kall), B): contiguous rows for the
    gathers of _trie_products."""
    rows = [br, 1.0 / br]
    if len(kall):
        rows.append((s_tot[:, None] ** (-0.5 * np.asarray(kall)[None, :])).astype(np.complex128))
    return np.ascontiguousarray(np.concatenate(rows, 1).T)


def evaluate_v6(st, P, scale=1.0):
    """|M|^2 of a version-6 structure: (1/4) sum over the surviving configurations of |A|^2 averaged with its
    parity image. `scale` multiplies the masses and widths, as in evaluate."""
    pl = _plan6(st)
    P = np.asarray(P, np.float64)
    res = np.zeros(P.shape[0])
    m_legs = [m * scale for m in st["m_legs"]]
    # events per chunk: a unit's products (rows x events) should stay in cache; measured optimum 8 at 2->6 (~9000 rows
    # per unit), >= 32 at 2->5 (~700)
    step = int(np.clip(70_000 // max(pl["nrows"] // max(len(pl["units"]), 1), 1), 4, 64))
    for lo in range(0, P.shape[0], step):
        Pc = P[lo:lo + step]
        dinv = _dinv(pl["base"], Pc, scale)
        br, s_tot = _spinor_brackets(Pc, m_legs)
        BT = _brtable(br, s_tot, pl["kall"])
        dinvT = np.ascontiguousarray(dinv.T)
        tot = np.zeros(len(Pc))
        for u in pl["units"]:
            U, V = _unit_UV(u, BT, dinvT)
            tot += u["w"] * (U.real ** 2 + U.imag ** 2 + V.real ** 2 + V.imag ** 2)
        res[lo:lo + step] = tot / 4.0
    return res


def helicity_amplitudes(st, P, lg=None):
    """Every surviving configuration's amplitude of a version-6 structure at momenta P (B, N, 4): {lambda: A (B,)},
    lambda as 2h per labelled leg (all-outgoing). A configuration not listed vanishes by the selection rules.
    lg (N,) applies the little-group scaling (_spinor_brackets)."""
    pl = _plan6(st)
    P = np.asarray(P, np.float64)
    dinv = _dinv(pl["base"], P, 1.0)
    br, s_tot = _spinor_brackets(P, st["m_legs"], lg)
    h = br.shape[1] // 2
    BT = _brtable(br, s_tot, pl["kall"])
    BS = _brtable(np.concatenate([br[:, h:], br[:, :h]], 1), s_tot, pl["kall"])   # every <ij> swapped with [ij]
    dinvT = np.ascontiguousarray(dinv.T)
    res = {}
    for u in pl["units"]:
        U, V = _unit_UV(u, BT, dinvT)
        res[u["lam"]] = U + 1j * V
        if u["w"] == 2.0:                                   # the partner: the same terms with brackets swapped
            U, V = _unit_UV(u, BS, dinvT)                   # (= U - iV of the above for physical spinors)
            res[_neg(u["lam"])] = U + 1j * V
    return res


def legal_diagram_count(pdgs, hgg=False):
    """Number of SM cubic tree diagrams (sm_vertices; ggH only on an external H) of the process pdgs (beams
    first, incoming), labelled legs: what version 3's trees are drawn from, the real count without the quartic
    vertices. E.g. e+e- -> mu+mu- 2, gg -> gg 3, u ubar -> Z + 4g 384, + 5g 3840. Subset recursion; the last leg is
    the root, so the ggH vertex is also legal at the top when that leg is the H (missing before 2026-10-12: it
    undercounted processes ending on an H, e.g. u ubar -> g Z H, which no number reported here was)."""
    tab = _line_table(hgg)
    N = len(pdgs)
    out = [_anti(p) if i < 2 else p for i, p in enumerate(pdgs)]
    top = frozenset(range(N - 1))

    @functools.lru_cache(maxsize=None)
    def f(S):                        # legs S -> {line X leaving S: number of subtrees}
        if len(S) == 1:
            return {out[next(iter(S))]: 1}
        res, legs = {}, sorted(S)
        for r in range(len(legs) - 1):
            for comb in itertools.combinations(legs[1:], r):
                A = frozenset((legs[0],) + comb)
                Bs = S - A
                for Xa, ca in f(A).items():
                    for Xb, cb in f(Bs).items():
                        for X, cls in tab.get((Xa, Xb), ()):
                            if cls != "HGG" or (Xa == 25 and len(A) == 1) or (Xb == 25 and len(Bs) == 1) \
                                    or (S == top and out[N - 1] == 25):
                                res[X] = res.get(X, 0) + ca * cb
        return res
    return f(top).get(_anti(out[N - 1]), 0)


def label_events(name, events):
    """Labeller for the generation path: events = [(momenta (N,4), pdg)], returns |M|^2."""
    return evaluate(structure(name), np.stack([e[0] for e in events]))


def catalog_entry(name):
    """mg5_pipeline_final.PROCESSES entry of a synthetic process."""
    st = structure(name)
    cfg = {
        "kind": "synthetic", "nfinal": st["nfinal"], "pdg_ids": list(st["pdg_ids"]),
        "m_finals": list(st["m_legs"][2:]), "param_card_patches": {},
        "alphas_power": st["k_s"], "order": list(st.get("order", [0, 0, st["k_s"], st["nfinal"] - st["k_s"]])),
        "internal_masses": dict(st["internal_masses"]), "gen_weight": 1.0,
        "synthetic": {"version": st["version"], "sha": st["sha"]}, "syn_name": name, "sqrts": list(st["sqrts"]),
        "mg5_generate": [],
    }
    if st["nfinal"] == 2:
        cfg["m_final"] = list(st["m_legs"][2:])
    return cfg


def sidecar(name):
    """The diagram sidecar payload (tools/dump_diagrams.py format) of a synthetic process:
    one subprocess, legs numbered by slot (1, 2 initial), every diagram and its images under
    the identical-leg permutations."""
    st = structure(name)
    pdgs, N = st["pdg_ids"], len(st["pdg_ids"])
    external = [{"number": i + 1, "pdg": int(pdgs[i]), "state": "in" if i < 2 else "out"} for i in range(N)]
    graphs, seen = [], set()
    for perm in st["perms"]:
        for d in st["diagrams"]:
            key = frozenset((tuple(sorted(perm[i] for i in p["legs"])), p["pdg"]) for p in d["props"])
            if key in seen:
                continue
            seen.add(key)
            nodes = [{"kind": "external", "interaction_id": 0, "leg_number": perm[i] + 1,
                      "pdg": int(pdgs[i]), "state": "in" if i < 2 else "out"} for i in range(N)]
            vert = {}                                     # cluster id -> vertex node creating it
            n_vertex = len(d["merges"]) + 1
            orders = ([dict(_ORDER_OF[c]) for c in d["vertex_class"]] if "vertex_class" in d else
                      [{"QCD": 1} if v in d["qcd_vertices"] else {"QED": 1} for v in range(n_vertex)])
            for v in range(n_vertex):
                nodes.append({"kind": "vertex", "interaction_id": -1, "orders": orders[v]})
            edges = []
            consumer = {}
            for v, (a, b) in enumerate(d["merges"]):
                vert[N + v] = N + v
                consumer[a] = consumer[b] = N + v
            for c in d["root"]:
                consumer[c] = N + len(d["merges"])
            for i in range(N):
                edges.append({"u": i, "v": consumer[i], "pdg": int(pdgs[i]), "external": True,
                              "leg_number": perm[i] + 1})
            for k, p in enumerate(d["props"]):
                c = N + k
                edges.append({"u": vert[c], "v": consumer[c], "pdg": int(p["pdg"]), "external": False,
                              "leg_number": 0})
            graphs.append({"nodes": nodes, "edges": edges, "diagram_type": "tree"})
    tag = " ".join(str(p) for p in pdgs[:2]) + " > " + " ".join(str(p) for p in pdgs[2:])
    return {"process": name, "model": f"synthetic_v{st['version']}", "mg5_generate": [],
            "synthetic_sha": st["sha"], "n_subprocesses": 1,
            "subprocesses": [{"process_str": f"synthetic {tag}", "external": external,
                              "n_diagrams": len(graphs), "diagrams": graphs}]}


def write_sidecar(name, out_dir=None):
    """Write ``<out_dir>/<name>.diagrams.json`` unless an up-to-date one is there (same
    structure hash); atomic, so concurrent trials cannot read a half-written file."""
    out_dir = out_dir or DEFAULT_DIAGRAMS_DIR
    path = os.path.join(out_dir, f"{name}.diagrams.json")
    sha = structure(name)["sha"]
    try:
        with open(path) as f:
            if json.load(f).get("synthetic_sha") == sha:
                return path
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    os.makedirs(out_dir, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(sidecar(name), f)
    os.replace(tmp, path)
    return path


def recipe_entries(names, n_train, n_val, n_test):
    return [{"name": nm, "sqrts": [int(structure(nm)["sqrts"][0]), int(structure(nm)["sqrts"][1])],
             "n_train": int(n_train), "n_val": int(n_val), "n_test": int(n_test)} for nm in names]


def _describe(name):
    st = structure(name)
    print(f"{name}  v{st['version']}  sha={st['sha']}  {st['pdg_ids'][:2]} -> {st['pdg_ids'][2:]}  m={st['m_legs'][2:]}"
          f"  K={len(st['diagrams'])}" + (f" of cap {st['k_cap']}" if "k_cap" in st else "")
          + f"  order={st.get('order', [0, 0, st['k_s'], st['nfinal'] - st['k_s']])}")
    hel = (f"configurations={len(st['configs'])}  numerators={len(st['numerators'])}" if st["version"] >= 6
           else f"H={len(st['numerators'])}")
    print(f"  sqrt(s) {st['sqrts']}  k_s={st['k_s']}  g={st['coupling']:.3e}  {hel}"
          f"  perms={len(st['perms'])}  scalar mass={st['internal_masses'][25]:.1f}")
    if st.get("revision") == 2 and "n_legal" in st:
        print(f"  revision 2: {st['n_legal']} legal diagrams, {st['k_all']} labelled kept"
              f"{' (subsampled)' if st['subsampled'] else ''} in {len(st['diagrams'])} permutation orbits")
    for d in st["diagrams"][:40]:
        print("  diagram: " + ", ".join(
            f"{p['pdg']}{p['legs']}(M={p['mass']:.1f},G={p['width']:.3g},e={p['power']})" for p in d["props"])
              + "".join(f"  B{sp['legs']}" for sp in d.get("spinors", [])))


def main():
    import argparse
    import yaml
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--describe", nargs="*", default=[])
    ap.add_argument("--recipe", help="write a recipe YAML listing syn_<first>..syn_<first+count-1>")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--sidecars", help="write the diagram sidecars of every syn* process of a recipe")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    for nm in a.describe:
        _describe(nm)
    if a.recipe:
        names = [f"syn_{k:05d}" for k in range(a.first, a.first + a.count)]
        doc = {"sampling": {"mode": "mixture"},
               "processes": recipe_entries(names, a.n_train, a.n_val, a.n_test)}
        head = (f"# Synthetic-amplitude prior: {a.count} synthetic processes syn_{a.first:05d}.. "
                f"(tools/synthetic_amplitudes.py, v{SYN_VERSION}),\n# mixture sampling as the ladder "
                f"recipes; sqrt(s) windows from each structure (max(25, 1.05 threshold), the catalog's rule).\n")
        with open(a.recipe, "w") as f:
            f.write(head)
            yaml.safe_dump(doc, f, default_flow_style=None, sort_keys=False, width=200)
        print(f"wrote {a.recipe} ({a.count} processes)")
    if a.sidecars:
        with open(a.sidecars) as f:
            doc = yaml.safe_load(f)
        procs = doc["processes"] if isinstance(doc, dict) else doc
        for p in procs:
            if is_synthetic(p["name"]):
                print(write_sidecar(p["name"], a.out_dir))


if __name__ == "__main__":
    main()
