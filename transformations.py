import torch
import numpy as np

def compute_invariants(particles, eps=1e-4, incl_diag_invariants=False):
    # compute matrix of all inner products
    def inner_product(p1, p2): return p1[..., 0] * p2[..., 0] - (
        p1[..., 1:] * p2[..., 1:]
    ).sum(dim=-1)
    if incl_diag_invariants:
        offset = 0
    else:
        offset = 1
    idxs = torch.triu_indices(
        particles.shape[-2], particles.shape[-2], offset=offset)
    invariants = inner_product(
        particles[..., idxs[0], :], particles[..., idxs[1], :])
    invariants = invariants.clamp(min=eps)
    

def log(x):
    return x.log()