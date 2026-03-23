import torch


class AmplitudeDataset(torch.utils.data.Dataset):
    def __init__(self, particles, amplitudes, tokens, dtype):
        self.particles  = [torch.tensor(p, dtype=dtype) for p in particles]
        self.amplitudes = [torch.tensor(a, dtype=dtype) for a in amplitudes]
        self.tokens     = [torch.tensor(t, dtype=torch.long) for t in tokens]
        self.len = min(len(p) for p in self.particles)

    def __len__(self):
        return self.len

    # dataset.py
    def __getitem__(self, idx):
        return [
            (particles[idx], amplitudes[idx], tokens[idx])
            for (particles, amplitudes, tokens) in zip(
                self.particles, self.amplitudes, self.tokens
            )
        ]
    
def collate_variable_length(batch):
    """
    batch: list of length batch_size, each element is a list of (particles, amplitude) 
        tuples, one per dataset.
    
    For LLoCA we DON'T pad — we return the raw variable-length tensors per dataset,
    because LLoCA's attention mask is built inside _batch_loss from the actual n_particles.
    """
    n_datasets = len(batch[0])
    result = []
    for idataset in range(n_datasets):
        particles = torch.stack([item[idataset][0] for item in batch], dim=0)
        amplitudes = torch.stack([item[idataset][1] for item in batch], dim=0)
        tokens = torch.stack([item[idataset][2] for item in batch], dim=0)
        result.append((particles, amplitudes, tokens))
    return result