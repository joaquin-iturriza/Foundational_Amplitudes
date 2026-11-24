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


def test_brute_gram_and_jackknife():
    # the brute check's record: its Gram matrix reproduces the summed terms, and the jackknife error over its M batches
    # covers the known answer of the linear-regression case
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis", "critical_batch"))
    from noise_read import gram_terms, jackknife_se
    torch.manual_seed(2)
    D, s, N, M = 10, 0.5, 64, 16
    w_star = torch.randn(D, dtype=torch.float64)
    d = 0.3 * torch.randn(D, dtype=torch.float64)
    w = torch.nn.Parameter(w_star + d)
    exp = SimpleNamespace(optimizer=torch.optim.SGD([w], lr=0.0))
    hits = 0
    for _ in range(40):
        xs = [torch.randn(N, D, dtype=torch.float64) for _ in range(M)]
        losses = ((0.5 * (x @ w - (x @ w_star + s * torch.randn(N, dtype=torch.float64))).square()).mean() for x in xs)
        k, terms, gram = BaseExperiment._noise_grads(exp, losses, {"all": [w]}, gram=True)
        r = {"k": k, "b_small": float(N), "b_big": float(N * k), "all_gram": gram["all"]}
        S, G2, _ = gram_terms(r, list(range(k)))
        S2, G22 = noise_scale_terms(*terms["all"], N, N * k)
        assert abs(S / S2 - 1) < 1e-9 and abs(G2 / G22 - 1) < 1e-9
        se = jackknife_se(r)
        S_true = D * (d.square().sum() + s ** 2) + d.square().sum()
        hits += abs(S - S_true) <= 2 * se[0]
    assert hits >= 32   # about 95% of 40 within two standard errors


def test_chunks_overstate_trace_under_per_process_means():
    # the loss is a mean over processes of per-process means, so a batch gradient is self-normalised: its spread
    # across batches has no term from the batch's process counts. Chunks weighted at the whole batch's counts do
    # carry that term (a chunk's share of each process varies), so the chunk tr Sigma is above the batch one
    # whenever the processes' mean gradients differ; with one process the two agree (FA-ccin2p3-225d38ea: 1.28x)
    import numpy as np

    def ratio(P, n, k=8, M=16, reps=150, D=5):
        rng = np.random.default_rng(0)
        mu = rng.normal(size=(P, D))
        Sc = Sb = 0.0
        for _ in range(reps):
            gb = []
            for _ in range(M):
                p = rng.integers(0, P, n)
                g = mu[p] + rng.normal(size=(n, D))
                cnt = np.bincount(p, minlength=P)
                w = 1.0 / ((cnt > 0).sum() * cnt[p])
                gb.append((w[:, None] * g).sum(0))
                ch = np.array([k * (w[c, None] * g[c]).sum(0) for c in np.array_split(np.arange(n), k)])
                Sc += noise_scale_terms((ch ** 2).sum(1).mean(), (ch.mean(0) ** 2).sum(), n / k, n)[0] / M
            gb = np.array(gb)
            Sb += noise_scale_terms((gb ** 2).sum(1).mean(), (gb.mean(0) ** 2).sum(), n, n * M)[0]
        return Sc / Sb

    assert abs(ratio(1, 512) - 1) < 0.05
    assert ratio(50, 600) > 1.5
