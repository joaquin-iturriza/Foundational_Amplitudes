import json
import os

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
        import numpy as np
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
        import numpy as np
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