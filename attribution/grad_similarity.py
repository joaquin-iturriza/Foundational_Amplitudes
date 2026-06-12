"""
GradSim: gradient similarity data attribution.

Computes cosine similarity between:
  g_target = E[∇L(target; θ)]   — target dataset gradient
  g_i      = E[∇L(d_i; θ_pre)] — pre-trained model on each source dataset

Attribution score per source dataset = cosine_similarity(g_target, g_i).

The checkpoint used for g_target (θ) is configurable:

  θ = θ_ft  (fine-tuned, recommended for full fine-tuning):
    Asks "which pre-training datasets have gradient alignment with where
    the fine-tuned model currently sits on the target loss surface?"
    Uses information from the fine-tuning process. Does NOT work for
    LoRA fine-tuning (different parameter names after LoRA injection).

  θ = θ_pre (pre-trained, required for LoRA fine-tuning):
    Asks "which pre-training datasets pushed the model in the direction
    the target task requires, starting from θ_pre?" Equivalent to
    evaluating the gradient alignment at the beginning of fine-tuning.

Reference: single-checkpoint case of TracIn (Pruthi et al., NeurIPS 2020,
arXiv:2002.08484), adapted for dataset-level attribution as in
LESS (Xia et al., ICML 2024, arXiv:2402.04333).
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Project root needed for shared imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataset import AmplitudeDataset, collate_variable_length, build_flat_arrays
from preprocessing import preprocess_amplitude
from particle_ids import ParticleTokenizer
from losses import LogCoshLoss, RelL1Loss, HeteroscedasticLoss


# ---------------------------------------------------------------------------
# Loss construction
# ---------------------------------------------------------------------------

def build_loss(loss_name: str) -> torch.nn.Module:
    """Return loss module matching the name used in the training config."""
    match loss_name:
        case "MSE":
            return torch.nn.MSELoss()
        case "L1":
            return torch.nn.L1Loss()
        case "LogCosh":
            return LogCoshLoss()
        case "RelL1":
            return RelL1Loss()
        case _:
            raise ValueError(
                f"Unsupported loss '{loss_name}'. "
                "Supported: MSE, L1, LogCosh, RelL1."
            )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _preprocess_lloca(particles_t: torch.Tensor):
    """Replicate init_data() LLoCa preprocessing.

    Boosts all events to their CoM frame then applies a random Lorentz rotation.
    Returns (particles_prepd as numpy array, mom_mean, mom_std).
    """
    from lloca.utils.rand_transforms import rand_lorentz
    from lloca.utils.polar_decomposition import restframe_boost

    m2 = particles_t[..., 0] ** 2 - (particles_t[..., 1:] ** 2).sum(dim=-1)
    particles_t[..., 0] = torch.sqrt(
        (particles_t[..., 1:] ** 2).sum(dim=-1) + m2.clamp(min=0)
    )
    lab = particles_t[..., :2, :].sum(dim=-2)
    to_com = restframe_boost(lab)
    trafo = rand_lorentz(particles_t.shape[:-2], generator=None, dtype=particles_t.dtype)
    trafo = torch.einsum("...ij,...jk->...ik", trafo, to_com)
    particles_t = torch.einsum("...ij,...kj->...ki", trafo, particles_t)

    prepd = particles_t / particles_t.std()
    return prepd.numpy(), float(prepd.mean()), float(prepd.std().clamp(min=1e-2))


def load_dataset(
    dataset_name: str,
    data_path: str,
    tokenizer: ParticleTokenizer,
    *,
    dtype: torch.dtype = torch.float32,
    n_events: int | None = None,
    amp_trafos: list | None = None,
) -> tuple[AmplitudeDataset, float, float]:
    """Load and preprocess one physics process dataset for LLoCa-style models.

    Parameters
    ----------
    dataset_name : str
        Dataset file stem (no .npy).
    data_path : str
        Directory containing .npy files.
    tokenizer : ParticleTokenizer
        Token encoder (should match the model whose gradients will be computed).
    dtype : torch.dtype
        Tensor dtype — must match model dtype.
    n_events : int | None
        Load only the first n_events rows. None = full dataset.
    amp_trafos : list | None
        Amplitude preprocessing transforms, e.g. ['log', 'standardization'].

    Returns
    -------
    dataset  : AmplitudeDataset
    mom_mean : float  — momentum normalization mean for LLoCa forward pass
    mom_std  : float  — momentum normalization std  for LLoCa forward pass
    """
    fpath = os.path.join(data_path, f"{dataset_name}.npy")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Dataset not found: {fpath}")

    raw = np.load(fpath)
    if n_events is not None:
        raw = raw[:n_events]

    n_particles  = (raw.shape[1] - 1) // 5
    momenta_cols = n_particles * 4

    particles  = raw[:, :momenta_cols]
    pdg_ids    = raw[:, momenta_cols:-1].astype(int)
    amplitudes = raw[:, [-1]]

    type_tokens = tokenizer.register_and_encode(pdg_ids)

    pt = torch.tensor(particles, dtype=torch.float64).reshape(-1, n_particles, 4)
    particles_prepd, mom_mean, mom_std = _preprocess_lloca(pt)

    amplitudes_prepd, _, _ = preprocess_amplitude(amplitudes, trafos=amp_trafos)

    parts_list  = [particles_prepd[j] for j in range(len(particles_prepd))]
    tokens_list = [type_tokens[j]     for j in range(len(type_tokens))]
    particles_flat, tokens_flat, offsets = build_flat_arrays(parts_list, tokens_list)

    return (
        AmplitudeDataset(
            particles_flat=particles_flat,
            offsets=offsets,
            amplitudes=amplitudes_prepd,
            tokens_flat=tokens_flat,
            dtype=dtype,
        ),
        mom_mean,
        mom_std,
    )


def _make_loader(dataset: AmplitudeDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_variable_length,
        pin_memory=torch.cuda.is_available() and num_workers > 0,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------

def _batch_loss(
    model: torch.nn.Module,
    batch: tuple,
    loss_fn: torch.nn.Module,
    mom_mean: float,
    mom_std: float,
    device: torch.device,
) -> torch.Tensor:
    """Forward pass + loss for one LLoCa batch."""
    particles, y, tokens, ptr = batch
    y_pred = model(
        particles.to(device), tokens.to(device),
        mean=mom_mean, std=mom_std,
        ptr=ptr.to(device),
    )
    return loss_fn(y_pred, y.to(device))


def _avg_gradient(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    mom_mean: float,
    mom_std: float,
    n_batches: int,
    device: torch.device,
) -> tuple[dict, int]:
    """Accumulate average gradient of the task loss over up to n_batches batches.

    Mirrors EWC._compute_fisher() in fine_tune.py but accumulates the raw gradient
    (not squared), giving the mean gradient direction rather than curvature.

    Returns (grad_dict, n_batches_used).
    """
    acc = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    model.eval()
    seen = 0
    for batch in loader:
        if seen >= n_batches:
            break
        loss = _batch_loss(model, batch, loss_fn, mom_mean, mom_std, device)
        model.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                acc[name].add_(param.grad.data)
        seen += 1
    for name in acc:
        acc[name].div_(max(seen, 1))
    model.zero_grad()
    return acc, seen


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def cosine_similarity(g1: dict, g2: dict) -> float:
    """Cosine similarity between two gradient dicts (flattened to 1D).

    Uses only keys common to both dicts, in g1's iteration order, so the
    vectors are always aligned regardless of dict ordering.
    """
    keys = [k for k in g1 if k in g2]
    v1 = torch.cat([g1[k].flatten() for k in keys])
    v2 = torch.cat([g2[k].flatten() for k in keys])
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def dot_product(g1: dict, g2: dict) -> float:
    """Dot product between two gradient dicts."""
    return sum((g1[k] * g2[k]).sum().item() for k in g1 if k in g2)


# ---------------------------------------------------------------------------
# Main attribution function
# ---------------------------------------------------------------------------

def run_gradsim(
    *,
    model_pre: torch.nn.Module,
    model_target: torch.nn.Module | None = None,
    loss_fn: torch.nn.Module,
    target_dataset: AmplitudeDataset,
    target_mom_mean: float,
    target_mom_std: float,
    source_datasets: dict,   # {name: (AmplitudeDataset, mom_mean, mom_std)}
    n_batches: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict:
    """Compute gradient similarity scores for each source dataset vs. the target.

    Parameters
    ----------
    model_pre    : pre-trained model — used for all source dataset gradients
    model_target : model used to compute g_target.  Pass the fine-tuned model
                   for full fine-tuning (recommended); pass None or model_pre
                   to use the pre-trained checkpoint (required for LoRA).
    loss_fn      : task loss (no regularization)
    source_datasets : {name: (AmplitudeDataset, mom_mean, mom_std)}
    n_batches    : max batches to average gradient over per dataset
    device       : torch device

    Returns
    -------
    dict with keys:
        'scores'              — {dataset_name: {'cosine': float, 'dot': float}}
        'attribution_ranking' — dataset names sorted by cosine, descending
    """
    if model_target is None:
        model_target = model_pre

    # Target gradient
    target_loader = _make_loader(target_dataset, batch_size, num_workers)
    g_target, n_target = _avg_gradient(
        model_target, target_loader, loss_fn,
        target_mom_mean, target_mom_std, n_batches, device,
    )
    print(f"[target] gradient computed over {n_target} batches", flush=True)

    # Source gradients at pre-trained checkpoint
    scores = {}
    for name, (ds, mm, ms) in source_datasets.items():
        loader = _make_loader(ds, batch_size, num_workers)
        g_i, n_seen = _avg_gradient(
            model_pre, loader, loss_fn, mm, ms, n_batches, device,
        )
        scores[name] = {
            "cosine": cosine_similarity(g_target, g_i),
            "dot":    dot_product(g_target, g_i),
        }
        print(
            f"[source] {name}: cosine={scores[name]['cosine']:+.4f}"
            f"  ({n_seen} batches)",
            flush=True,
        )

    ranking = sorted(scores, key=lambda k: scores[k]["cosine"], reverse=True)
    return {"scores": scores, "attribution_ranking": ranking}
