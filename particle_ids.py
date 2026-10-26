import json
import os
import numpy as np

class ParticleTokenizer:
    """
    Maps PDG IDs to contiguous embedding indices.
    Indices are assigned in the order particles are first registered.
    Index 0 is reserved for padding.
    """
    def __init__(self):
        self._pdg_to_idx = {}   # e.g. {21: 1, 22: 2, 23: 3, ...}

    def register(self, pdg_id: int) -> int:
        """Add a PDG ID if not seen before. Returns its index."""
        if pdg_id not in self._pdg_to_idx:
            self._pdg_to_idx[pdg_id] = len(self._pdg_to_idx) + 1  # 0 reserved for padding
        return self._pdg_to_idx[pdg_id]

    def encode(self, pdg_ids):
        """
        pdg_ids: array-like of ints, shape (n_events, n_particles)
        returns: same shape, values replaced by embedding indices
        raises KeyError if an unknown PDG ID is encountered at inference time
        """
        pdg_ids = np.asarray(pdg_ids, dtype=int)
        result = np.zeros_like(pdg_ids)
        for pdg_id in np.unique(pdg_ids):
            if pdg_id not in self._pdg_to_idx:
                raise KeyError(
                    f"PDG ID {pdg_id} not in tokenizer. "
                    f"Call register() during training data loading first."
                )
            result[pdg_ids == pdg_id] = self._pdg_to_idx[pdg_id]
        return result

    def register_and_encode(self, pdg_ids):
        """Convenience: register all IDs then encode. Use during training only."""
        pdg_ids = np.asarray(pdg_ids, dtype=int)
        for pdg_id in np.unique(pdg_ids):
            self.register(pdg_id)
        return self.encode(pdg_ids)

    @property
    def vocab_size(self) -> int:
        """Total number of known particle types + 1 for padding."""
        return len(self._pdg_to_idx) + 1

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({int(k): v for k, v in self._pdg_to_idx.items()}, f)

    @classmethod
    def load(cls, path: str) -> "ParticleTokenizer":
        obj = cls()
        with open(path, "r") as f:
            obj._pdg_to_idx = {int(k): v for k, v in json.load(f).items()}
        return obj


# ---------------------------------------------------------------------------
# Physical property table
# ---------------------------------------------------------------------------
# Feature vector (9 numbers):
#   [charge, spin, log10_mass_gev, weak_isospin_t3, baryon_number,
#    lepton_number, color_charge, color_casimir, generation]
#
#   charge          — electric charge Q (e.g. -1, +2/3, 0)
#   spin            — spin quantum number (0, 0.5, 1)
#   log10_mass_gev  — log10(m/GeV) OF THE MASS THE GENERATOR USES (the MadGraph
#                     sm card with restrict_default plus the pipeline's locked
#                     values, GENERATOR_MASSES_GEV below): u, d, s, c, e, mu and
#                     the neutrinos are massless there and get _MASSLESS = -5.0;
#                     tau 1.777, b 4.7, t 172.5 (mg5_pipeline_final.LOCKED_MT),
#                     Z 91.188, W 80.419, H 125. PDG values do not belong here:
#                     the table must describe the particles in the events (it
#                     used to carry PDG masses, so d and s, both generated
#                     massless, differed by a label the data could not support).
#   weak_isospin_t3 — T₃, third component of weak SU(2)_L isospin
#                     quarks: ±1/2 (u-type +1/2, d-type -1/2)
#                     leptons: -1/2 (charged), +1/2 (neutrino)
#                     W±: ±1;  Z, γ, g, H: 0
#                     antiparticles: sign flipped
#   baryon_number   — B (1/3 quarks, -1/3 antiquarks, 0 else)
#                     explicit here so leptoquarks (B≠0 and L≠0) are correct
#   lepton_number   — L (+1 leptons, -1 antileptons, 0 else)
#   color_charge    — sign of SU(3) fundamental charge:
#                     +1 triplet (quark), -1 antitriplet (antiquark), 0 else
#   color_casimir   — quadratic Casimir C₂(R) of the SU(3) representation:
#                     0 singlet, 4/3 fundamental (3 or 3̄), 3 adjoint (8),
#                     10/3 sextet, … generalises to any color rep.
#   generation      — fermion generation 1, 2, 3 (u d e νe / c s μ νμ / t b τ ντ);
#                     0 for bosons. This is what separates d from s and u from c
#                     when both are massless: the momenta carry no flavour, and
#                     same-generation pairs (ud, cs) have W exchange that
#                     cross-generation pairs (us) lack, so without it the model
#                     sees ud→ud and us→us, dd→dd and ds→ds as one input with two
#                     targets (docs/results.tex, catalog_v2 census). Categorical:
#                     one-hot it (generation_onehot) like spin and color.
#
# These 9 numbers identify every SM particle as the generator sees it:
#   γ vs Z    — by log10_mass_gev
#   γ vs g    — by color_casimir (0 vs 3)
#   e vs μ vs τ — by generation (and τ by mass)
#   u vs c vs t — by generation (and t by mass)
#   quark vs antiquark — by sign of charge, baryon_number, color_charge
#   gluino vs gluon — by spin (0.5 vs 1) and log10_mass_gev
#
# WHY project through a fixed hidden dim rather than feed directly:
#   Adding a quantum number changes n_features but NOT d_particle_hidden,
#   so the transformer architecture (and all its weights) is unchanged.
#   Only the tiny projection matrix Linear(n_features → d_particle_hidden) needs
#   to be extended — its new column is initialized to zero so the model's
#   existing predictions are unaffected before any fine-tuning.

_MASSLESS = -5.0   # sentinel for log10(mass/GeV); well below electron (-3.3)

# Masses the generator actually uses (GeV): MadGraph sm, restrict_default, plus
# the pipeline's locked heavy-quark values. Keep in sync with
# mg5_pipeline_final.LOCKED_MT / TOP_PATCHES and the m_b card default (4.7);
# tests/test_particle_encoding.py checks it.
GENERATOR_MASSES_GEV = {
    "tau": 1.777, "b": 4.7, "t": 172.5, "Z": 91.188, "W": 80.419, "H": 125.0,
}
_LM = {k: float(np.log10(v)) for k, v in GENERATOR_MASSES_GEV.items()}

N_PARTICLE_FEATURES = 9
PARTICLE_FEATURE_NAMES = [
    "charge", "spin", "log10_mass_gev", "weak_isospin_t3",
    "baryon_number", "lepton_number", "color_charge", "color_casimir", "generation",
]

PARTICLE_PROPERTIES = {
    # fmt: off
    # ── gauge bosons ──────────────────────────────────────────────────────────
    #         Q     s   log10m       T₃    B     L   col  C₂   gen
    21: [  0.0,  1.0,  _MASSLESS,  0.0,  0.0,  0.0,  0.0,  3.0,  0.0 ],  # gluon
    22: [  0.0,  1.0,  _MASSLESS,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 ],  # photon
    23: [  0.0,  1.0,  _LM["Z"],   0.0,  0.0,  0.0,  0.0,  0.0,  0.0 ],  # Z
    24: [  1.0,  1.0,  _LM["W"],   1.0,  0.0,  0.0,  0.0,  0.0,  0.0 ],  # W+
   -24: [ -1.0,  1.0,  _LM["W"],  -1.0,  0.0,  0.0,  0.0,  0.0,  0.0 ],  # W-
    # ── Higgs ─────────────────────────────────────────────────────────────────
    25: [  0.0,  0.0,  _LM["H"],   0.0,  0.0,  0.0,  0.0,  0.0,  0.0 ],  # H
    # ── quarks (u, d, s, c massless in the generator) ─────────────────────────
     1: [ -1/3,  0.5,  _MASSLESS,  -0.5,  1/3,  0.0,  1.0,  4/3,  1.0 ],  # d
    -1: [  1/3,  0.5,  _MASSLESS,   0.5, -1/3,  0.0, -1.0,  4/3,  1.0 ],  # dbar
     2: [  2/3,  0.5,  _MASSLESS,   0.5,  1/3,  0.0,  1.0,  4/3,  1.0 ],  # u
    -2: [ -2/3,  0.5,  _MASSLESS,  -0.5, -1/3,  0.0, -1.0,  4/3,  1.0 ],  # ubar
     3: [ -1/3,  0.5,  _MASSLESS,  -0.5,  1/3,  0.0,  1.0,  4/3,  2.0 ],  # s
    -3: [  1/3,  0.5,  _MASSLESS,   0.5, -1/3,  0.0, -1.0,  4/3,  2.0 ],  # sbar
     4: [  2/3,  0.5,  _MASSLESS,   0.5,  1/3,  0.0,  1.0,  4/3,  2.0 ],  # c
    -4: [ -2/3,  0.5,  _MASSLESS,  -0.5, -1/3,  0.0, -1.0,  4/3,  2.0 ],  # cbar
     5: [ -1/3,  0.5,  _LM["b"],   -0.5,  1/3,  0.0,  1.0,  4/3,  3.0 ],  # b
    -5: [  1/3,  0.5,  _LM["b"],    0.5, -1/3,  0.0, -1.0,  4/3,  3.0 ],  # bbar
     6: [  2/3,  0.5,  _LM["t"],    0.5,  1/3,  0.0,  1.0,  4/3,  3.0 ],  # t
    -6: [ -2/3,  0.5,  _LM["t"],   -0.5, -1/3,  0.0, -1.0,  4/3,  3.0 ],  # tbar
    # ── charged leptons (e, mu massless in the generator) ─────────────────────
    11: [ -1.0,  0.5,  _MASSLESS,  -0.5,  0.0,  1.0,  0.0,  0.0,  1.0 ],  # e-
   -11: [  1.0,  0.5,  _MASSLESS,   0.5,  0.0, -1.0,  0.0,  0.0,  1.0 ],  # e+
    13: [ -1.0,  0.5,  _MASSLESS,  -0.5,  0.0,  1.0,  0.0,  0.0,  2.0 ],  # μ-
   -13: [  1.0,  0.5,  _MASSLESS,   0.5,  0.0, -1.0,  0.0,  0.0,  2.0 ],  # μ+
    15: [ -1.0,  0.5,  _LM["tau"], -0.5,  0.0,  1.0,  0.0,  0.0,  3.0 ],  # τ-
   -15: [  1.0,  0.5,  _LM["tau"],  0.5,  0.0, -1.0,  0.0,  0.0,  3.0 ],  # τ+
    # ── neutrinos (massless; T₃ = +½ for ν_L) ────────────────────────────────
    12: [  0.0,  0.5,  _MASSLESS,   0.5,  0.0,  1.0,  0.0,  0.0,  1.0 ],  # νe
   -12: [  0.0,  0.5,  _MASSLESS,  -0.5,  0.0, -1.0,  0.0,  0.0,  1.0 ],  # ν̄e
    14: [  0.0,  0.5,  _MASSLESS,   0.5,  0.0,  1.0,  0.0,  0.0,  2.0 ],  # νμ
   -14: [  0.0,  0.5,  _MASSLESS,  -0.5,  0.0, -1.0,  0.0,  0.0,  2.0 ],  # ν̄μ
    16: [  0.0,  0.5,  _MASSLESS,   0.5,  0.0,  1.0,  0.0,  0.0,  3.0 ],  # ντ
   -16: [  0.0,  0.5,  _MASSLESS,  -0.5,  0.0, -1.0,  0.0,  0.0,  3.0 ],  # ν̄τ
    # fmt: on
}

# ── non-SM particles (D49; docs/results.tex sec:nonsm) ─────────────────────────
# APPEND ONLY, in a dict of their own: GLOBAL_PDG_IDX numbers the SM entries above first
# (sorted, exactly as before these rows existed) and these after them in insertion order,
# so every SM particle keeps its index and its encoded row. A checkpoint's property_matrix
# buffer is then the first rows of the current one (wrappers.AmplitudeLLoCaWrapper loads
# it as such), and a pretrained SM model reads a new particle with no retraining.
# Masses: the generator's, i.e. the parameter point the recipes generate
# (mg5_pipeline_final.NONSM_POINT), one mass per PDG id as for the SM rows.
# Quantum numbers: T3 of the field's upper/lower doublet component for the charged scalar
# (H+ = +1/2, H- = -1/2, as the SM table gives the W its T3 = +-1), 0 for the neutral
# scalars like the SM H, 0 for the Z'-like vector (an SU(2) singlet).
NONSM_MASSES_GEV = {"Y1": 500.0, "H+": 200.0, "h2": 250.0, "h3": 300.0}
BSM_PARTICLE_PROPERTIES = {
    # fmt: off
    #               Q     s   log10m                               T3    B    L   col  C2   gen
    5000001: [  0.0,  1.0,  float(np.log10(NONSM_MASSES_GEV["Y1"])),  0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],  # Y1 (DMsimp_s_spin1 vector mediator)
         37: [  1.0,  0.0,  float(np.log10(NONSM_MASSES_GEV["H+"])),  0.5, 0.0, 0.0, 0.0, 0.0, 0.0 ],  # H+ (2HDM)
        -37: [ -1.0,  0.0,  float(np.log10(NONSM_MASSES_GEV["H+"])), -0.5, 0.0, 0.0, 0.0, 0.0, 0.0 ],  # H- (2HDM)
         35: [  0.0,  0.0,  float(np.log10(NONSM_MASSES_GEV["h2"])),  0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],  # h2 (2HDM, CP-even)
         36: [  0.0,  0.0,  float(np.log10(NONSM_MASSES_GEV["h3"])),  0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],  # h3 (2HDM, CP-odd)
    # fmt: on
}
# ── heavy-quark species (catalog_v3, D102): the top's quantum numbers at the masses its heavy-quark layer is generated
# at, so a pool generated at m_t = 300 GeV reads its top as a 300 GeV particle (the table's masses are the generator's).
# Codes sign * (6000000 + m); a pool's PDG 6 is mapped to them by species_pdgs() from its recipe's physics.masses.
HEAVY_QUARK_MASSES_GEV = (50, 300, 600)
for _m in HEAVY_QUARK_MASSES_GEV:
    for _sgn in (1, -1):
        _row = list(PARTICLE_PROPERTIES[6 * _sgn]); _row[2] = float(np.log10(_m))
        BSM_PARTICLE_PROPERTIES[_sgn * (6000000 + _m)] = _row
# ── the massless b (catalog_v4, D130: the five-flavour scheme, MadGraph's sm-no_b_mass, b massless in beams and finals):
# the b's quantum numbers at zero mass, so a catalog_v4 pool reads its b as the massless particle it was generated as while
# catalog_v3's pools (b at 4.7 GeV) keep the row above. Code sign * (5000000 + m) with m = 0, the heavy-quark layer's rule;
# a pool's PDG 5 is mapped to it by species_pdgs() from its process entry's generated b mass (b_mass_of_process).
MASSLESS_B_PDG = 5000000
for _sgn in (1, -1):
    _row = list(PARTICLE_PROPERTIES[5 * _sgn]); _row[2] = _MASSLESS
    BSM_PARTICLE_PROPERTIES[_sgn * MASSLESS_B_PDG] = _row
SM_PDGS = tuple(sorted(PARTICLE_PROPERTIES))      # frozen before the merge below
assert not set(BSM_PARTICLE_PROPERTIES) & set(SM_PDGS)
PARTICLE_PROPERTIES = {**PARTICLE_PROPERTIES, **BSM_PARTICLE_PROPERTIES}


class ParticleFeaturizer:
    """Maps PDG IDs to fixed physical property vectors."""

    N_FEATURES = N_PARTICLE_FEATURES
    FEATURE_NAMES = PARTICLE_FEATURE_NAMES

    def get_feature_vector(self, pdg_id: int) -> np.ndarray:
        if pdg_id not in PARTICLE_PROPERTIES:
            raise KeyError(
                f"PDG ID {pdg_id} not in PARTICLE_PROPERTIES. "
                f"Add it to particle_ids.py with its quantum numbers."
            )
        return np.array(PARTICLE_PROPERTIES[pdg_id], dtype=np.float32)


# ---------------------------------------------------------------------------
# Global fixed PDG → property-table index  (use_PIDs=False mode)
# ---------------------------------------------------------------------------
# The mapping is sorted by PDG ID for reproducibility and is independent of
# which particles appear in any particular training dataset.  This is the key
# property that enables zero-shot BSM inference: a new particle added to
# PARTICLE_PROPERTIES above gets an index here at import time — no retraining
# of the transformer needed, only (optionally) fine-tuning the projection layer.

_sorted_pdgs     = list(SM_PDGS) + list(BSM_PARTICLE_PROPERTIES)   # SM sorted, then non-SM appended (see above)
GLOBAL_PDG_IDX   = {pdg: i + 1 for i, pdg in enumerate(_sorted_pdgs)}  # 1-based; 0 = padding
GLOBAL_N_ENTRIES = len(_sorted_pdgs) + 1

_mat = np.zeros((GLOBAL_N_ENTRIES, N_PARTICLE_FEATURES), dtype=np.float32)
for _pdg, _idx in GLOBAL_PDG_IDX.items():
    _mat[_idx] = PARTICLE_PROPERTIES[_pdg]
GLOBAL_PROPERTY_MATRIX = _mat   # (GLOBAL_N_ENTRIES, N_FEATURES), row 0 = padding zeros

# The value add_is_massless_flag gives massless rows: the mean log10 mass over the massive
# SM species, from the SM rows only (float32 like the table), so non-SM rows never shift it.
_mcol = PARTICLE_FEATURE_NAMES.index("log10_mass_gev")
_sm_m = np.array([PARTICLE_PROPERTIES[p][_mcol] for p in SM_PDGS], dtype=np.float32)
MASSLESS_NEUTRAL_LOG10 = np.float32(_sm_m[np.abs(_sm_m - _MASSLESS) >= 1e-6].mean())


# ---------------------------------------------------------------------------
# Optional hybrid encoding: one-hot the *categorical* spin column
# ---------------------------------------------------------------------------
# Spin is a label, not a magnitude — spin-1 is not "twice" spin-½, and spin-½
# is not "halfway between" 0 and 1.  Fed as a raw scalar through Linear(n→d),
# every spin is forced onto one line through the origin (contributions
# 0·w, 0.5·w, 1·w), so the projection can't give bosons and fermions
# independent embedding directions.  One-hot gives each spin value its own
# learnable row instead.  We keep the additive/continuous quantum numbers
# (charge, T₃, B, L, mass, casimir) as scalars — for those the ordering and
# distance ARE physical, so one-hot would throw information away.
#
# Pre-allocated spin values get a clean one-hot; anything outside the list
# falls back to a scalar "overflow" channel, so you do NOT have to pick a hard
# maximum spin up front — arbitrarily high/unusual spins still encode (you just
# lose the one-hot benefit for those rare cases until you add them to the list).
SPIN_ONEHOT_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0)   # extend freely; regen base shapes
_SPIN_TOL = 1e-6
_SPIN_COL = PARTICLE_FEATURE_NAMES.index("spin")   # == 1


def expand_spin_onehot(matrix, names=None):
    """Replace the scalar spin column of a property matrix with a one-hot over
    SPIN_ONEHOT_VALUES plus a scalar overflow channel.

    Returns (expanded_matrix, expanded_feature_names).  Layout is the original
    column order with the single spin column expanded in place:
        [charge, spin_is_0, spin_is_0.5, …, spin_overflow, <rest scalars…>]
    All-zero rows (the padding row 0) are preserved as all-zero so padded
    particles still contribute nothing.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    n      = matrix.shape[0]
    names  = list(names) if names is not None else list(PARTICLE_FEATURE_NAMES)
    spin   = matrix[:, _SPIN_COL]

    onehot   = np.zeros((n, len(SPIN_ONEHOT_VALUES)), dtype=np.float32)
    overflow = np.zeros((n, 1), dtype=np.float32)
    matched  = np.zeros(n, dtype=bool)
    for i, v in enumerate(SPIN_ONEHOT_VALUES):
        hit         = np.abs(spin - v) < _SPIN_TOL
        onehot[:, i] = hit
        matched     |= hit
    overflow[:, 0] = np.where(matched, 0.0, spin)   # 0 for known spins, raw value otherwise

    expanded = np.concatenate(
        [matrix[:, :_SPIN_COL], onehot, overflow, matrix[:, _SPIN_COL + 1:]],
        axis=1,
    )
    # keep padding rows all-zero (spin 0.0 would otherwise one-hot to a real slot)
    padding = ~np.any(matrix != 0.0, axis=1)
    expanded[padding] = 0.0

    names = (
        names[:_SPIN_COL]
        + [f"spin_is_{v:g}" for v in SPIN_ONEHOT_VALUES]
        + ["spin_overflow"]
        + names[_SPIN_COL + 1:]
    )
    return expanded, names


# ---------------------------------------------------------------------------
# Optional encoding: one-hot the categorical SU(3) color representation
# ---------------------------------------------------------------------------
# Same lesson as spin: the color rep is a *categorical label*, not a magnitude.
# `color_casimir` ∈ {0, 4/3, 3, 10/3, …} tags the rep (singlet / fundamental /
# adjoint / sextet); fed as a raw scalar through Linear(n→d) every rep is forced
# onto one line through the origin, so the octet and the triplet can't get
# independent embedding directions ("adjoint = 2.25×fundamental" is not a
# meaningful distance). We one-hot the rep identity — distinguishing 3 from 3̄ by
# the sign of `color_charge` — and REPLACE the scalar `color_charge`/`color_casimir`
# columns (exactly as `expand_spin_onehot` replaces the scalar spin), with a single
# `color_overflow` scalar that carries the Casimir of any rep outside the listed
# set so unlisted reps stay distinguishable.
#
# Each entry is (color_charge, color_casimir); extend freely (regen base shapes).
COLOR_REP_VALUES = (
    ( 0.0, 0.0  ),   # singlet  (leptons, γ, Z, W, H)
    ( 1.0, 4/3  ),   # triplet  3   (quark)
    (-1.0, 4/3  ),   # antitriplet 3̄ (antiquark)
    ( 0.0, 3.0  ),   # adjoint  8   (gluon)
)
_COLOR_TOL = 1e-6


def expand_color_onehot(matrix, names=None):
    """Replace the scalar ``color_charge``/``color_casimir`` columns with a one-hot
    over COLOR_REP_VALUES (rep identity, including 3 vs 3̄) plus a scalar overflow
    channel carrying the Casimir of any *unlisted* rep.

    This mirrors :func:`expand_spin_onehot` exactly — the categorical scalars are
    REMOVED, not kept alongside, so there is no one-hot/scalar redundancy. The
    one-hot slots already encode the 3-vs-3̄ sign (separate triplet/antitriplet
    rows), so dropping ``color_charge`` loses nothing; the overflow preserves
    magnitude for reps outside the list. Returns ``(matrix, feature_names)``;
    padding rows (all-zero) stay all-zero.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    names  = list(names) if names is not None else list(PARTICLE_FEATURE_NAMES)
    cc = names.index("color_charge")
    assert names[cc + 1] == "color_casimir", \
        "expand_color_onehot expects color_charge immediately followed by color_casimir"
    c2 = cc + 1

    n        = matrix.shape[0]
    onehot   = np.zeros((n, len(COLOR_REP_VALUES)), dtype=np.float32)
    overflow = np.zeros((n, 1), dtype=np.float32)
    matched  = np.zeros(n, dtype=bool)
    for i, (q_col, casimir) in enumerate(COLOR_REP_VALUES):
        hit = (np.abs(matrix[:, cc] - q_col) < _COLOR_TOL) & \
              (np.abs(matrix[:, c2] - casimir) < _COLOR_TOL)
        onehot[:, i] = hit
        matched     |= hit
    overflow[:, 0] = np.where(matched, 0.0, matrix[:, c2])  # Casimir for unlisted reps

    expanded = np.concatenate(
        [matrix[:, :cc], onehot, overflow, matrix[:, c2 + 1:]], axis=1)
    # keep padding rows all-zero (singlet (0,0) would otherwise one-hot to a slot)
    padding = ~np.any(matrix != 0.0, axis=1)
    expanded[padding] = 0.0

    names_out = (names[:cc]
                 + [f"color_is_{i}" for i in range(len(COLOR_REP_VALUES))]
                 + ["color_overflow"]
                 + names[c2 + 1:])
    return expanded, names_out


# ---------------------------------------------------------------------------
# Optional hybrid encoding: one-hot the *categorical* generation column
# ---------------------------------------------------------------------------
# Generation is a label (the third generation is not "three times" the first),
# and what the model needs from it is equality between two legs (same-generation
# pairs exchange a W). One learnable direction per generation; bosons are all-zero.
GENERATION_VALUES = (1.0, 2.0, 3.0)


def expand_generation_onehot(matrix, names=None):
    """Replace the scalar ``generation`` column by ``gen_is_1..3`` one-hot columns
    (bosons all zero). Returns ``(matrix, names)``; padding rows stay zero."""
    matrix = np.asarray(matrix, dtype=np.float32)
    names  = list(names) if names is not None else list(PARTICLE_FEATURE_NAMES)
    g = names.index("generation")
    onehot = np.zeros((matrix.shape[0], len(GENERATION_VALUES)), dtype=np.float32)
    for i, v in enumerate(GENERATION_VALUES):
        onehot[:, i] = (np.abs(matrix[:, g] - v) < 1e-6)
    expanded = np.concatenate([matrix[:, :g], matrix[:, g + 1:], onehot], axis=1)
    padding = ~np.any(matrix != 0.0, axis=1)
    expanded[padding] = 0.0
    names_out = names[:g] + names[g + 1:] + [f"gen_is_{int(v)}" for v in GENERATION_VALUES]
    return expanded, names_out


# ---------------------------------------------------------------------------
# Optional encoding: explicit "exactly massless" flag
# ---------------------------------------------------------------------------
# `log10_mass_gev` uses a _MASSLESS = -5.0 sentinel, which (a) conflates the
# *categorical* fact "symmetry-protected exactly massless" (γ, g, ν) with "very
# light", and (b) sits as a large outlier far below the electron (-3.3), so it
# dominates the single Linear's init/gradients. We split the two: a binary
# ``is_massless`` flag carries the categorical fact, and the massless rows'
# scalar mass is neutralised to the mean over massive species so the scalar
# axis carries magnitude only (≈0 after standardization).
def add_is_massless_flag(matrix, names=None):
    """Append a binary ``is_massless`` column and neutralise the massless
    sentinel in ``log10_mass_gev``. Returns ``(matrix, names)``; padding stays 0."""
    matrix = np.asarray(matrix, dtype=np.float32).copy()
    names  = list(names) if names is not None else list(PARTICLE_FEATURE_NAMES)
    m      = names.index("log10_mass_gev")

    padding  = ~np.any(matrix != 0.0, axis=1)
    massless = (np.abs(matrix[:, m] - _MASSLESS) < 1e-6) & ~padding
    massive  = ~massless & ~padding

    flag = massless.astype(np.float32).reshape(-1, 1)
    # neutralise sentinel → mean log10 mass of the massive SM species, frozen (a non-SM row
    # must not move the value every massless particle is encoded with)
    if massless.any():
        matrix[massless, m] = MASSLESS_NEUTRAL_LOG10

    expanded = np.concatenate([matrix, flag], axis=1)
    return expanded, names + ["is_massless"]


# ---------------------------------------------------------------------------
# Optional: per-column standardization of the *continuous* features
# ---------------------------------------------------------------------------
# The raw columns live on wildly different scales (log10_mass −5…+2 vs charge
# ±1), so through one Xavier Linear the mass column dominates the initial
# embedding and gradients. Z-score the continuous columns across the real
# (non-padding) species. Categorical / one-hot / binary columns are left as-is
# (standardizing them would break their 0/1 sparsity and the all-zero padding).
_STD_SKIP_PREFIXES = ("spin_is_", "spin_overflow", "color_is_", "color_overflow", "is_massless", "gen_is_")


# FROZEN standardization constants (per physical feature), computed ONCE over the
# current SM particle table. They are hardcoded — NOT recomputed from the table —
# so that extending PARTICLE_PROPERTIES with a new species does NOT shift the
# encoding of existing particles, preserving the project's zero-shot-new-particle
# property (a pretrained encoder stays valid; a new particle is just placed on the
# frozen scale). `log10_mass_gev` stats are over the is_massless-neutralised values
# (the massless sentinel replaced by the massive mean), matching the
# prop_is_massless pipeline that runs before standardize.
#
# Regenerated when the table changed to the generator's masses and gained the
# generation column (catalog_v2 census, docs/results.tex). Regenerate via:
#   mat, names = add_is_massless_flag(GLOBAL_PROPERTY_MATRIX)
#   real = np.any(GLOBAL_PROPERTY_MATRIX != 0, axis=1)
#   {f: (mat[real, names.index(f)].mean(), mat[real, names.index(f)].std()) for f in CONT}
_FROZEN_STD_STATS = {
    "charge":          (0.00000000, 0.61463630),
    "spin":            (0.56666666, 0.21343747),
    "log10_mass_gev":  (1.41847146, 0.46235374),   # generator masses; massless rows neutralised
    "weak_isospin_t3": (0.00000000, 0.51639777),
    "baryon_number":   (0.00000000, 0.21081853),
    "lepton_number":   (0.00000000, 0.63245553),
    "color_charge":    (0.00000000, 0.63245553),
    "color_casimir":   (0.63333333, 0.78102499),
    "generation":      (1.60000002, 1.08320510),   # used only when generation_onehot is off
}


def standardize_property_columns(matrix, names=None):
    """Z-score continuous columns using FROZEN per-feature constants
    (``_FROZEN_STD_STATS``), so the encoding is invariant to which particles are
    in the table. Categorical/one-hot/binary columns are skipped; padding rows
    stay zero. A continuous feature not present in the frozen table falls back to
    table statistics (with a warning) so unusual configs still run.
    Returns ``(matrix, names)`` (names unchanged)."""
    matrix = np.asarray(matrix, dtype=np.float32).copy()
    names  = list(names) if names is not None else list(PARTICLE_FEATURE_NAMES)
    padding = ~np.any(matrix != 0.0, axis=1)
    real    = ~padding
    for j, name in enumerate(names):
        if name.startswith(_STD_SKIP_PREFIXES):
            continue
        if name in _FROZEN_STD_STATS:
            mu, sd = _FROZEN_STD_STATS[name]
        else:
            col = matrix[real, j]
            mu, sd = float(col.mean()), float(col.std())
            import warnings
            warnings.warn(f"standardize_property_columns: no frozen stats for "
                          f"'{name}', falling back to table statistics.")
        matrix[real, j] = (matrix[real, j] - mu) / sd if sd > 1e-12 else (matrix[real, j] - mu)
    matrix[padding] = 0.0
    return matrix, names


def drop_generation(matrix, names=None):
    """Remove the ``generation`` column: the encoding as it was before the column
    existed (d ≡ s, u ≡ c once the masses are the generator's). For A/B baselines
    only (``data.generation_feature: false``)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    names  = list(names) if names is not None else list(PARTICLE_FEATURE_NAMES)
    g = names.index("generation")
    return np.concatenate([matrix[:, :g], matrix[:, g + 1:]], axis=1), names[:g] + names[g + 1:]


def build_property_matrix(spin_onehot=False, color_onehot=False,
                          is_massless=False, standardize=False,
                          generation_onehot=False, generation_feature=True):
    """Assemble the particle property matrix with the requested smart-encoding
    transforms, applied in a fixed order (spin → color → generation → mass-flag →
    standardize). Returns ``(matrix, feature_names)``. With all flags False this
    is exactly ``(GLOBAL_PROPERTY_MATRIX, PARTICLE_FEATURE_NAMES)``;
    ``generation_feature=False`` drops the generation column altogether."""
    mat, names = GLOBAL_PROPERTY_MATRIX, list(PARTICLE_FEATURE_NAMES)
    if not generation_feature:
        mat, names = drop_generation(mat, names)
        generation_onehot = False
    if spin_onehot:
        mat, names = expand_spin_onehot(mat, names)
    if color_onehot:
        mat, names = expand_color_onehot(mat, names)
    if generation_onehot:
        mat, names = expand_generation_onehot(mat, names)
    if is_massless:
        mat, names = add_is_massless_flag(mat, names)
    if standardize:
        mat, names = standardize_property_columns(mat, names)
    return mat, names


# ---------------------------------------------------------------------------
# Per-particle (data-derived) on-shell mass
# ---------------------------------------------------------------------------
# The table's ``log10_mass_gev`` is a frozen per-species constant. When
# ``data.mass_from_momenta`` is set we instead REPLACE that column, per particle,
# with the on-shell mass derived from the event 4-momentum (m=sqrt(E^2-|p|^2)) —
# exact for external legs, Lorentz-invariant, and faithful to whatever mass the
# dataset was actually generated with. The model (wrappers) does the per-particle
# computation in the forward pass; this helper returns the small metadata needed
# to reproduce the *same* downstream transform the table column would have had
# under the active smart-encoding flags (massless neutralisation → standardize),
# so the data-derived value lands on the identical scale.
#
# Threshold: log10(m/GeV) below MASS_MASSLESS_LOG10_THR is treated as "exactly
# massless" (symmetry-protected: gamma/g/nu, or light quarks generated massless).
# It sits below the electron (-3.29) and above the _MASSLESS sentinel (-5).
MASS_MASSLESS_LOG10_THR = -4.0


def mass_feature_spec(spin_onehot=False, color_onehot=False,
                      is_massless=False, standardize=False, generation_onehot=False,
                      generation_feature=True):
    """Metadata for the data-derived mass column (see ``data.mass_from_momenta``).

    Returns a dict the wrapper uses to map a per-particle physical mass onto the
    same scale the frozen ``log10_mass_gev`` column would occupy under the active
    flags::

        mass_col       : int   column index of log10_mass_gev in the built matrix
        is_massless_col: int|None  column index of the is_massless flag, if present
        floor_log10    : float massless sentinel value (_MASSLESS); the floor of log10(m)
        threshold_log10: float log10(m) below this ⇒ treated as exactly massless
        neutralize_log10: float|None  value massless rows take when the is_massless
                          flag is active (mean log10 over massive species); None ⇒
                          massless rows keep the sentinel
        std_mu, std_sd : float|None  z-score constants if standardize is active, else None
    """
    _, names = build_property_matrix(
        spin_onehot=spin_onehot, color_onehot=color_onehot,
        is_massless=is_massless, standardize=standardize,
        generation_onehot=generation_onehot, generation_feature=generation_feature,
    )
    mass_col = names.index("log10_mass_gev")
    is_massless_col = names.index("is_massless") if "is_massless" in names else None

    neutralize_log10 = None
    if is_massless:
        # Reproduce add_is_massless_flag's neutralisation: massless rows take the
        # mean log10 over the massive species (computed on the raw table).
        neutralize_log10 = float(MASSLESS_NEUTRAL_LOG10)

    std_mu = std_sd = None
    if standardize:
        std_mu, std_sd = _FROZEN_STD_STATS["log10_mass_gev"]

    return {
        "mass_col":        int(mass_col),
        "is_massless_col": is_massless_col,
        "floor_log10":     float(_MASSLESS),
        "threshold_log10": float(MASS_MASSLESS_LOG10_THR),
        "neutralize_log10": neutralize_log10,
        "std_mu":          None if std_mu is None else float(std_mu),
        "std_sd":          None if std_sd is None else float(std_sd),
    }


def b_mass_of_process(cfg, physics=None):
    """The mass (GeV) the generator gave the external b legs of process entry `cfg` (an mg5_pipeline_final.PROCESSES
    entry, its recipe `physics` applied), None when it has no external b. A b beam has the entry's `beam_masses` (absent:
    the pipeline's BEAM_MASS, the b on its 4.7 GeV shell; catalog_v4's `beam_masses: {}` makes every beam massless), a
    final b its `m_finals` (or `m_final`); a physics.masses {5: m} wins over both. The legs of one entry must agree."""
    pdg = [int(p) for p in cfg["pdg_ids"]]
    pm = {abs(int(k)): float(v) for k, v in (((physics or {}).get("masses")) or {}).items()}
    if 5 in pm:
        return pm[5] if any(abs(p) == 5 for p in pdg) else None
    bm = cfg.get("beam_masses")
    bm = {5: GENERATOR_MASSES_GEV["b"]} if bm is None else {abs(int(k)): float(v) for k, v in bm.items()}
    if "m_finals" in cfg:
        mf = [float(x) for x in cfg["m_finals"]]
    else:
        mf = cfg.get("m_final", 0.0)
        mf = [float(x) for x in mf] if isinstance(mf, (list, tuple)) else [float(mf)] * (len(pdg) - 2)
    ms = [bm.get(5, 0.0) for p in pdg[:2] if abs(p) == 5] + [m for p, m in zip(pdg[2:], mf) if abs(p) == 5]
    if not ms:
        return None
    assert max(ms) - min(ms) < 1e-9, f"{cfg.get('mg5_generate')}: b legs generated at different masses {ms}"
    return ms[0]


def species_pdgs(pdg_ids, physics=None, process=None):
    """The PDG ids the property table reads for a pool generated with recipe `physics` from process entry `process`
    (its mg5_pipeline_final.PROCESSES entry): a top generated at a heavy-quark mass (physics.masses {6: m}, m in
    HEAVY_QUARK_MASSES_GEV) becomes that mass's species; a b generated massless (catalog_v4, b_mass_of_process == 0)
    becomes the massless b (MASSLESS_B_PDG); anything else is unchanged. Without `process` the b is left as is.
    For the encoding only: the pools, the slot order, the order vectors and the target keep the generated ids."""
    if process is not None and b_mass_of_process(process, physics) == 0.0:
        pdg_ids = np.array(pdg_ids, dtype=int, copy=True)
        pdg_ids[pdg_ids == 5] = MASSLESS_B_PDG; pdg_ids[pdg_ids == -5] = -MASSLESS_B_PDG
    m = ((physics or {}).get("masses") or {})
    m = {abs(int(k)): float(v) for k, v in m.items()}.get(6)
    # a top mass with no species row (catalog_v2's ee_ttbar__mt185 interpolation probe) keeps the table's 172.5, as
    # every run on it so far has read it
    if m is None or abs(m - round(m)) > 1e-9 or int(round(m)) not in HEAVY_QUARK_MASSES_GEV:
        return pdg_ids
    pdg_ids = np.array(pdg_ids, dtype=int, copy=True)
    pdg_ids[pdg_ids == 6] = 6000000 + int(round(m)); pdg_ids[pdg_ids == -6] = -(6000000 + int(round(m)))
    return pdg_ids


def global_encode(pdg_ids: np.ndarray) -> np.ndarray:
    """Map PDG IDs → global property-table indices (use_PIDs=False mode)."""
    pdg_ids = np.asarray(pdg_ids, dtype=int)
    result  = np.zeros_like(pdg_ids)
    for pdg in np.unique(pdg_ids):
        if pdg not in GLOBAL_PDG_IDX:
            raise KeyError(
                f"PDG ID {pdg} not in PARTICLE_PROPERTIES. "
                f"Add it with its quantum numbers to particle_ids.py — "
                f"the trained model can then process it at inference immediately."
            )
        result[pdg_ids == pdg] = GLOBAL_PDG_IDX[pdg]
    return result
