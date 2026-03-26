import numpy as np
import torch
from torch.utils.data import Dataset
 
 
class AmplitudeDataset(Dataset):
    """
    Single flat dataset produced after padding and concatenating all source
    files in init_data.  Each item is a 3-tuple:
        particles  : float tensor  (nparticles_max, 4)        [LLOCA]
                     or            (nparticles_max * 4,)      [other models]
        amplitude  : float tensor  (1,)
        tokens     : long tensor   (nparticles_max,)
                     real particles carry their encoded PDG token,
                     padded slots carry 0 (the reserved padding index)
    """
 
    def __init__(self, particles, amplitudes, tokens, dtype, reshape=False):
        """
        Parameters
        ----------
        particles  : np.ndarray  (N, nparticles_max * 4)
        amplitudes : np.ndarray  (N, 1)
        tokens     : np.ndarray  (N, nparticles_max)  int
        dtype      : torch dtype
        reshape    : if True particles are returned as (nparticles_max, 4)
        """
        nparticles_max = tokens.shape[1]
        self.particles  = torch.tensor(particles,  dtype=dtype)
        self.amplitudes = torch.tensor(amplitudes, dtype=dtype)
        self.tokens     = torch.tensor(tokens,     dtype=torch.long)
        if reshape:
            self.particles = self.particles.view(-1, nparticles_max, 4)
 
    def __len__(self):
        return len(self.amplitudes)
 
    def __getitem__(self, idx):
        return (self.particles[idx], self.amplitudes[idx], self.tokens[idx])
 
 
def collate_variable_length(batch):
    """
    batch: list of (particles, amplitude, tokens) tuples.
    Returns a single (particles, amplitudes, tokens) tuple of stacked tensors.
    The name is kept for drop-in compatibility with the existing DataLoader calls.
    """
    particles  = torch.stack([item[0] for item in batch], dim=0)
    amplitudes = torch.stack([item[1] for item in batch], dim=0)
    tokens     = torch.stack([item[2] for item in batch], dim=0)
    return particles, amplitudes, tokens