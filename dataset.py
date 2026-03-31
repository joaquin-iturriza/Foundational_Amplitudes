import numpy as np
import torch
from torch.utils.data import Dataset
 
 
class AmplitudeDataset(Dataset):
    """
    Single flat dataset. Each item is a 3-tuple:
        particles  : float tensor  (n_particles, 4)   — natural multiplicity, no padding
        amplitude  : float tensor  (1,)
        tokens     : long tensor   (n_particles,)
 
    Events within a batch have different n_particles. collate_variable_length
    handles this by concatenating them into a flat sparse representation with
    a ptr tensor, matching the interface expected by LLoCa's framesnet and
    backbone when ptr is not None.
    """
 
    def __init__(self, particles_list, amplitudes, tokens_list, dtype):
        """
        Parameters
        ----------
        particles_list : list of np.ndarray, each shape (n_particles_i, 4)
        amplitudes     : np.ndarray  (N, 1)
        tokens_list    : list of np.ndarray, each shape (n_particles_i,)
        dtype          : torch dtype
        """
        self.particles  = [torch.tensor(p, dtype=dtype)      for p in particles_list]
        self.amplitudes = torch.tensor(amplitudes, dtype=dtype)
        self.tokens     = [torch.tensor(t, dtype=torch.long) for t in tokens_list]
 
    def __len__(self):
        return len(self.amplitudes)
 
    def __getitem__(self, idx):
        return (self.particles[idx], self.amplitudes[idx], self.tokens[idx])
 
 
def collate_variable_length(batch):
    """
    Collate a list of (particles, amplitude, tokens) tuples into a sparse batch.
 
    Returns
    -------
    particles  : (N_total, 4)    flat concatenation of all particles in the batch
    amplitudes : (B, 1)          one amplitude per event
    tokens     : (N_total,)      flat concatenation of all token sequences
    ptr        : (B+1,)          ptr[i] = start index of event i in the flat sequence
                                 e.g. for batch of 3 events with 4,4,3 particles:
                                 ptr = [0, 4, 8, 11]
    """
    particles_list  = [item[0] for item in batch]
    amplitudes_list = [item[1] for item in batch]
    tokens_list     = [item[2] for item in batch]
 
    particles  = torch.cat(particles_list,   dim=0)   # (N_total, 4)
    amplitudes = torch.stack(amplitudes_list, dim=0)  # (B, 1)
    tokens     = torch.cat(tokens_list,      dim=0)   # (N_total,)
 
    counts = torch.tensor([p.shape[0] for p in particles_list], dtype=torch.long)
    ptr    = torch.zeros(len(batch) + 1, dtype=torch.long)
    torch.cumsum(counts, dim=0, out=ptr[1:])
 
    return particles, amplitudes, tokens, ptr