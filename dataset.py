import numpy as np
import torch
from torch.utils.data import Dataset
 
 
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
 