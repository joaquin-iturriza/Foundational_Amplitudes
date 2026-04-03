import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
 
 
class AmplitudeDataset(Dataset):
    """
    Sparse variable-length dataset backed by contiguous numpy arrays for fast indexing.
 
    Particles are stored as one big (N_total_particles, 4) array plus an offsets
    array of shape (N_events, 2) where offsets[i] = [start, end] for event i.
    This makes __getitem__ O(1) with a simple slice, avoiding the Python list
    overhead of storing N_events separate arrays.
 
    Each item is a 3-tuple:
        particles : float tensor  (n_particles_i, 4)
        amplitude : float tensor  (1,)
        tokens    : long tensor   (n_particles_i,)
    """
 
    def __init__(self, particles_flat, offsets, amplitudes, tokens_flat, dtype):
        """
        Parameters
        ----------
        particles_flat : np.ndarray  (N_total_particles, 4)
        offsets        : np.ndarray  (N_events, 2)  — [start, end] per event
        amplitudes     : np.ndarray  (N_events, 1)
        tokens_flat    : np.ndarray  (N_total_particles,)
        dtype          : torch dtype
        """
        self.particles_flat = torch.tensor(particles_flat, dtype=dtype)
        self.offsets        = offsets                                      # keep as numpy for slicing
        self.amplitudes     = torch.tensor(amplitudes,     dtype=dtype)
        self.tokens_flat    = torch.tensor(tokens_flat,    dtype=torch.long)
 
    def __len__(self):
        return len(self.amplitudes)
 
    def __getitem__(self, idx):
        start, end = int(self.offsets[idx, 0]), int(self.offsets[idx, 1])
        return (
            self.particles_flat[start:end],   # (n_particles_i, 4)
            self.amplitudes[idx],              # (1,)
            self.tokens_flat[start:end],       # (n_particles_i,)
        )
 
 
def build_flat_arrays(particles_list, tokens_list):
    """
    Convert a list of per-event arrays into a contiguous flat array + offsets.
 
    Parameters
    ----------
    particles_list : list of np.ndarray, each (n_particles_i, 4)
    tokens_list    : list of np.ndarray, each (n_particles_i,)
 
    Returns
    -------
    particles_flat : (N_total_particles, 4)
    tokens_flat    : (N_total_particles,)
    offsets        : (N_events, 2)  int64
    """
    lengths   = np.array([p.shape[0] for p in particles_list], dtype=np.int64)
    starts    = np.concatenate([[0], np.cumsum(lengths[:-1])])
    ends      = starts + lengths
    offsets   = np.stack([starts, ends], axis=1)             # (N, 2)
 
    particles_flat = np.concatenate(particles_list, axis=0)  # (N_total, 4)
    tokens_flat    = np.concatenate(tokens_list,    axis=0)  # (N_total,)
 
    return particles_flat, tokens_flat, offsets
 
 
def collate_variable_length(batch):
    """
    Collate (particles, amplitude, tokens) tuples into a sparse batch.
 
    Returns
    -------
    particles  : (N_total, 4)   flat concatenation across all events in the batch
    amplitudes : (B, 1)
    tokens     : (N_total,)
    ptr        : (B+1,)         ptr[i] = start of event i in the flat tensors
    """
    particles_list  = [item[0] for item in batch]
    amplitudes_list = [item[1] for item in batch]
    tokens_list     = [item[2] for item in batch]
 
    particles  = torch.cat(particles_list,    dim=0)
    amplitudes = torch.stack(amplitudes_list, dim=0)
    tokens     = torch.cat(tokens_list,       dim=0)
 
    counts = torch.tensor([p.shape[0] for p in particles_list], dtype=torch.long)
    ptr    = torch.zeros(len(batch) + 1, dtype=torch.long)
    torch.cumsum(counts, dim=0, out=ptr[1:])
 
    return particles, amplitudes, tokens, ptr


class ProcessBalancedSampler(Sampler):
    """
    Yields batches where each process contributes equally (or according to
    dynamically updated weights).

    At each step, draws `batch_size // n_processes` indices from each process,
    then shuffles them together to form one batch.  When a process runs out of
    indices it reshuffles its own pool — so no process is ever exhausted.

    Weights are updated via `set_weights(weights)` where weights is a list of
    non-negative floats (one per process).  A weight of 0 disables that process.
    Internally weights are normalised to allocate slots per batch.

    Parameters
    ----------
    process_ids : np.ndarray  (N_events,)
        Integer process index for every event in the dataset (after shuffling).
    batch_size : int
    weights : list[float] | None
        Initial sampling weights per process.  Defaults to uniform.
    seed : int
    """

    def __init__(self, process_ids, batch_size, weights=None, seed=0):
        self.process_ids = np.asarray(process_ids)
        self.batch_size  = batch_size
        self.seed        = seed
        self.rng         = np.random.default_rng(seed)

        # indices per process
        unique = sorted(set(self.process_ids.tolist()))
        self.n_processes = len(unique)
        self.proc_indices = {
            p: np.where(self.process_ids == p)[0] for p in unique
        }
        # shuffled pools (refilled on exhaustion)
        self._pools = {p: self._shuffle(p) for p in unique}
        self._pool_pos = defaultdict(int)

        if weights is None:
            weights = [1.0] * self.n_processes
        self._weights = np.array(weights, dtype=float)

        # total number of batches per "epoch" based on smallest process
        min_proc_size = min(len(v) for v in self.proc_indices.values())
        self._n_batches = max(1, min_proc_size * self.n_processes // batch_size)

    def _shuffle(self, p):
        idx = self.proc_indices[p].copy()
        self.rng.shuffle(idx)
        return idx

    def set_weights(self, weights):
        """Update sampling weights (called after each validation)."""
        w = np.array(weights, dtype=float)
        w = np.clip(w, 0, None)
        self._weights = w

    def _slots_per_process(self):
        """Allocate batch slots proportionally to weights, at least 1 per active process."""
        w = self._weights.copy()
        total = w.sum()
        if total == 0:
            w = np.ones(self.n_processes, dtype=float)
            total = float(self.n_processes)
        slots = np.floor(w / total * self.batch_size).astype(int)
        # ensure at least 1 slot per active process
        slots = np.maximum(slots, (w > 0).astype(int))
        # distribute remaining slots to highest-weight processes
        remainder = self.batch_size - slots.sum()
        if remainder > 0:
            order = np.argsort(-w)
            for i in order[:remainder]:
                slots[i] += 1
        return slots

    def _next(self, p, n):
        """Draw n indices from process p, refilling pool as needed."""
        out = []
        while len(out) < n:
            pos   = self._pool_pos[p]
            pool  = self._pools[p]
            avail = len(pool) - pos
            need  = n - len(out)
            if avail >= need:
                out.extend(pool[pos:pos + need].tolist())
                self._pool_pos[p] = pos + need
            else:
                out.extend(pool[pos:].tolist())
                self._pools[p]    = self._shuffle(p)
                self._pool_pos[p] = 0
        return out

    def __iter__(self):
        for _ in range(self._n_batches):
            slots   = self._slots_per_process()
            procs   = sorted(self.proc_indices.keys())
            batch   = []
            for p, n in zip(procs, slots):
                if n > 0:
                    batch.extend(self._next(p, n))
            self.rng.shuffle(batch)
            yield from batch

    def __len__(self):
        return self._n_batches * self.batch_size
