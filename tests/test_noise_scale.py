"""The gradient noise scale (find_BS): base_experiment.noise_scale_terms and _noise_grads against a case whose answer is
known, and experiment._aggregate_per_process_loss's full_counts (the chunk weights of _noise_chunk_losses)."""
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from base_experiment import BaseExperiment, noise_scale_terms
from experiment import AmplitudeExperiment


def test_linear_regression_known_answer():
    # per-example loss (x.w - y)^2 / 2, x ~ N(0, I_D), y = x.w* + eps, eps ~ N(0, s^2); at w, with d = w - w*:
    # G = d, tr Sigma = D (|d|^2 + s^2) + |d|^2, so B_simple = tr Sigma / |G|^2 is known exactly
    torch.manual_seed(0)
    D, s, N, k, reps = 10, 0.5, 512, 8, 400
    w_star = torch.randn(D, dtype=torch.float64)
    d = 0.3 * torch.randn(D, dtype=torch.float64)
    w = torch.nn.Parameter(w_star + d)
    exp = SimpleNamespace(optimizer=torch.optim.SGD([w], lr=0.0))
    sets = {"all": [w]}
    S_sum = G2_sum = 0.0
    for _ in range(reps):
        x = torch.randn(N, D, dtype=torch.float64)
        y = x @ w_star + s * torch.randn(N, dtype=torch.float64)
        chunks = ((0.5 * (xc @ w - yc).square()).mean() for xc, yc in zip(x.chunk(k), y.chunk(k)))
        kk, terms = BaseExperiment._noise_grads(exp, chunks, sets)
        S, G2 = noise_scale_terms(*terms["all"], N / kk, N)
        S_sum, G2_sum = S_sum + S, G2_sum + G2
    S_true = D * (d.square().sum() + s ** 2) + d.square().sum()
    G2_true = d.square().sum()
    assert abs(S_sum / reps / S_true - 1) < 0.05
    assert abs(G2_sum / reps / G2_true - 1) < 0.05
    assert abs((S_sum / G2_sum) / (S_true / G2_true) - 1) < 0.07


def test_chunks_average_to_the_batch_loss():
    # mean aggregation over present processes of per-process means: k chunks at the batch's own counts, each times
    # n / chunk size, average to the batch loss although a chunk misses processes or holds them unevenly
    torch.manual_seed(1)
    stub = SimpleNamespace(cfg=OmegaConf.create({"training": {"loss": "MSE"}}), n_datasets=4,
                           _per_event_loss=lambda yp, y, sigma=None: (yp - y).square().reshape(-1))
    n, k = 24, 4
    pids = torch.tensor([0] * 15 + [1] * 6 + [2] * 3)[torch.randperm(n)]   # process 3 absent
    yp, y = torch.randn(n, 1), torch.randn(n, 1)
    full = AmplitudeExperiment._aggregate_per_process_loss(stub, yp, y, pids, "mean")
    counts = torch.bincount(pids, minlength=4)
    parts = [AmplitudeExperiment._aggregate_per_process_loss(stub, yp[c], y[c], pids[c], "mean", full_counts=counts)
             for c in torch.arange(n).chunk(k)]
    assert torch.allclose(sum(k * p for p in parts) / k, full)
