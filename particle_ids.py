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
# Feature vector (8 numbers):
#   [charge, spin, log10_mass_gev, weak_isospin_t3, baryon_number,
#    lepton_number, color_charge, color_casimir]
#
#   charge          — electric charge Q (e.g. -1, +2/3, 0)
#   spin            — spin quantum number (0, 0.5, 1)
#   log10_mass_gev  — log10(m/GeV); massless particles get _MASSLESS = -5.0
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
#
# These 8 numbers uniquely identify every SM particle, including:
#   γ vs Z    — by log10_mass_gev
#   γ vs g    — by color_casimir (0 vs 3)
#   e vs μ vs τ — by log10_mass_gev
#   u vs c vs t — by log10_mass_gev
#   quark vs antiquark — by sign of charge, baryon_number, color_charge
#   gluino vs gluon — by spin (0.5 vs 1) and log10_mass_gev
#
# WHY project through a fixed hidden dim rather than feed directly:
#   Adding an 9th quantum number changes n_features but NOT d_particle_hidden,
#   so the transformer architecture (and all its weights) is unchanged.
#   Only the tiny projection matrix Linear(n_features → d_particle_hidden) needs
#   to be extended — its new column is initialized to zero so the model's
#   existing predictions are unaffected before any fine-tuning.

_MASSLESS = -5.0   # sentinel for log10(mass/GeV); well below electron (-3.3)

N_PARTICLE_FEATURES = 8
PARTICLE_FEATURE_NAMES = [
    "charge", "spin", "log10_mass_gev", "weak_isospin_t3",
    "baryon_number", "lepton_number", "color_charge", "color_casimir",
]

PARTICLE_PROPERTIES = {
    # fmt: off
    # ── gauge bosons ──────────────────────────────────────────────────────────
    #         Q     s   log10m       T₃    B     L   col  C₂
    21: [  0.0,  1.0,  _MASSLESS,  0.0,  0.0,  0.0,  0.0,  3.0  ],  # gluon
    22: [  0.0,  1.0,  _MASSLESS,  0.0,  0.0,  0.0,  0.0,  0.0  ],  # photon
    23: [  0.0,  1.0,  np.log10(91.1876),  0.0,  0.0,  0.0,  0.0,  0.0  ],  # Z
    24: [  1.0,  1.0,  np.log10(80.377),   1.0,  0.0,  0.0,  0.0,  0.0  ],  # W+
   -24: [ -1.0,  1.0,  np.log10(80.377),  -1.0,  0.0,  0.0,  0.0,  0.0  ],  # W-
    # ── Higgs ─────────────────────────────────────────────────────────────────
    25: [  0.0,  0.0,  np.log10(125.25),   0.0,  0.0,  0.0,  0.0,  0.0  ],  # H
    # ── quarks (T₃: up-type +½, down-type -½) ────────────────────────────────
     1: [ -1/3,  0.5,  np.log10(4.67e-3),  -0.5,  1/3,  0.0,  1.0,  4/3 ],  # d
    -1: [  1/3,  0.5,  np.log10(4.67e-3),   0.5, -1/3,  0.0, -1.0,  4/3 ],  # dbar
     2: [  2/3,  0.5,  np.log10(2.16e-3),   0.5,  1/3,  0.0,  1.0,  4/3 ],  # u
    -2: [ -2/3,  0.5,  np.log10(2.16e-3),  -0.5, -1/3,  0.0, -1.0,  4/3 ],  # ubar
     3: [ -1/3,  0.5,  np.log10(9.34e-2),  -0.5,  1/3,  0.0,  1.0,  4/3 ],  # s
    -3: [  1/3,  0.5,  np.log10(9.34e-2),   0.5, -1/3,  0.0, -1.0,  4/3 ],  # sbar
     4: [  2/3,  0.5,  np.log10(1.27),      0.5,  1/3,  0.0,  1.0,  4/3 ],  # c
    -4: [ -2/3,  0.5,  np.log10(1.27),     -0.5, -1/3,  0.0, -1.0,  4/3 ],  # cbar
     5: [ -1/3,  0.5,  np.log10(4.18),     -0.5,  1/3,  0.0,  1.0,  4/3 ],  # b
    -5: [  1/3,  0.5,  np.log10(4.18),      0.5, -1/3,  0.0, -1.0,  4/3 ],  # bbar
     6: [  2/3,  0.5,  np.log10(172.76),    0.5,  1/3,  0.0,  1.0,  4/3 ],  # t
    -6: [ -2/3,  0.5,  np.log10(172.76),   -0.5, -1/3,  0.0, -1.0,  4/3 ],  # tbar
    # ── charged leptons (T₃ = -½ for ℓ_L, +½ for ℓ̄_L) ──────────────────────
    11: [ -1.0,  0.5,  np.log10(5.11e-4),  -0.5,  0.0,  1.0,  0.0,  0.0 ],  # e-
   -11: [  1.0,  0.5,  np.log10(5.11e-4),   0.5,  0.0, -1.0,  0.0,  0.0 ],  # e+
    13: [ -1.0,  0.5,  np.log10(1.057e-1), -0.5,  0.0,  1.0,  0.0,  0.0 ],  # μ-
   -13: [  1.0,  0.5,  np.log10(1.057e-1),  0.5,  0.0, -1.0,  0.0,  0.0 ],  # μ+
    15: [ -1.0,  0.5,  np.log10(1.777),    -0.5,  0.0,  1.0,  0.0,  0.0 ],  # τ-
   -15: [  1.0,  0.5,  np.log10(1.777),     0.5,  0.0, -1.0,  0.0,  0.0 ],  # τ+
    # ── neutrinos (massless; T₃ = +½ for ν_L) ────────────────────────────────
    12: [  0.0,  0.5,  _MASSLESS,           0.5,  0.0,  1.0,  0.0,  0.0 ],  # νe
   -12: [  0.0,  0.5,  _MASSLESS,          -0.5,  0.0, -1.0,  0.0,  0.0 ],  # ν̄e
    14: [  0.0,  0.5,  _MASSLESS,           0.5,  0.0,  1.0,  0.0,  0.0 ],  # νμ
   -14: [  0.0,  0.5,  _MASSLESS,          -0.5,  0.0, -1.0,  0.0,  0.0 ],  # ν̄μ
    16: [  0.0,  0.5,  _MASSLESS,           0.5,  0.0,  1.0,  0.0,  0.0 ],  # ντ
   -16: [  0.0,  0.5,  _MASSLESS,          -0.5,  0.0, -1.0,  0.0,  0.0 ],  # ν̄τ
    # fmt: on
}
# To add a BSM particle: append one entry here with its quantum numbers.
# The trained model's projection layer can then process it immediately at
# inference (the transformer weights are unchanged).  If you also want the
# model to *learn* the new particle's dynamics, fine-tune on a small BSM
# dataset — only the tiny particle_encoder layer needs updating.


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

_sorted_pdgs     = sorted(PARTICLE_PROPERTIES.keys())
GLOBAL_PDG_IDX   = {pdg: i + 1 for i, pdg in enumerate(_sorted_pdgs)}  # 1-based; 0 = padding
GLOBAL_N_ENTRIES = len(_sorted_pdgs) + 1

_mat = np.zeros((GLOBAL_N_ENTRIES, N_PARTICLE_FEATURES), dtype=np.float32)
for _pdg, _idx in GLOBAL_PDG_IDX.items():
    _mat[_idx] = PARTICLE_PROPERTIES[_pdg]
GLOBAL_PROPERTY_MATRIX = _mat   # (GLOBAL_N_ENTRIES, N_FEATURES), row 0 = padding zeros


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
