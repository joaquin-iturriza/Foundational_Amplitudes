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


def test_batch_accumulation_is_the_batch_gradient():
    # training.batch_accumulation: k micro-batches at the whole batch's counts, (chunk loss / k) backward each, give
    # the gradient of the one batch's loss (mean over present processes of per-process means, plus L2)
    torch.manual_seed(3)
    lin = torch.nn.Linear(4, 1).double()
    stub = SimpleNamespace(cfg=OmegaConf.create({"training": {"loss": "MSE"}}), n_datasets=5,
                           _per_event_loss=lambda yp, y, sigma=None: (yp - y).square().reshape(-1))
    n, k, lam = 40, 4, 1e-3
    x, y = torch.randn(n, 4, dtype=torch.float64), torch.randn(n, 1, dtype=torch.float64)
    pids = torch.randint(0, 4, (n,))   # process 4 absent
    reg = lambda: lam * sum(p.square().sum() for p in lin.parameters())
    agg = AmplitudeExperiment._aggregate_per_process_loss
    lin.zero_grad()
    (agg(stub, lin(x), y, pids, "mean") + reg()).backward()
    g_full = [p.grad.clone() for p in lin.parameters()]
    lin.zero_grad()
    counts = torch.bincount(pids, minlength=5)
    for c in torch.arange(n).chunk(k):
        lc = k * agg(stub, lin(x[c]), y[c], pids[c], "mean", full_counts=counts) + reg()
        (lc / k).backward()
    assert all(torch.allclose(a, p.grad, atol=1e-10) for a, p in zip(g_full, lin.parameters()))
    # the train mu-MSE on this path: the chunks' event-weighted mean of their MSEs is the batch's MSE
    with torch.no_grad():
        mses = [(torch.nn.functional.mse_loss(lin(x[c]), y[c]), len(c)) for c in torch.arange(n).chunk(k)]
        assert torch.isclose(sum(m * w for m, w in mses) / n, torch.nn.functional.mse_loss(lin(x), y))


def test_stop_at_val_loss():
    from base_experiment import BaseExperiment
    stub = lambda v: SimpleNamespace(cfg=OmegaConf.create({"training": {"stop_at_val_loss": v}}))
    assert BaseExperiment._reached_stop(stub(-2.3), -2.31) and BaseExperiment._reached_stop(stub(-2.3), -2.3)
    assert not BaseExperiment._reached_stop(stub(-2.3), -2.29) and not BaseExperiment._reached_stop(stub(None), -9.0)
    assert not BaseExperiment._reached_stop(stub(-2.3), None) and not BaseExperiment._reached_stop(stub(-2.3), float("nan"))


def test_update_map_is_the_optimizer_step():
    # Muon matrices: the map equals a real MuonAdamW step at momentum 0; AdamW parameters: g / (sqrt(v_hat) + eps) x lr
    from optim.muon import MuonAdamW
    torch.manual_seed(3)
    W = torch.nn.Parameter(torch.randn(6, 4, dtype=torch.float64))
    bvec = torch.nn.Parameter(torch.randn(4, dtype=torch.float64))
    opt = MuonAdamW([W, bvec], lr=0.01, momentum=0.0, nesterov=False, muon_weight_decay=0.0, mup=False,
                    muon_filter=lambda p: p.ndim == 2)
    exp = SimpleNamespace(optimizer=opt, model=torch.nn.Module(),
                          cfg=OmegaConf.create({"training": {"clip_grad_norm": None, "clip_grad_value": None}}))
    sets = {"muon": [W], "adamw": [bvec]}
    loss = lambda: (W.sum(1).square().sum() + (bvec * torch.arange(4.0, dtype=torch.float64)).square().sum())
    opt.zero_grad(); loss().backward(); opt.step()                 # one step: AdamW gets its second moment
    assert BaseExperiment._update_map(exp, [torch.zeros_like(W), torch.zeros_like(bvec)], sets) is not None
    opt.zero_grad(); loss().backward()
    g = [W.grad.clone(), bvec.grad.clone()]
    u = BaseExperiment._update_map(exp, g, sets)
    st = opt._adamw.state[bvec]
    v_after = st["exp_avg_sq"] * 0.999 + 0.001 * g[1].square()     # AdamW's moment after the next step's update
    W0 = W.detach().clone()
    opt.step()
    assert torch.allclose(W0 - W.detach(), u[0], rtol=1e-4, atol=1e-7)
    vhat = st["exp_avg_sq"].double() / (1 - 0.999 ** float(st["step"]))
    assert torch.allclose(st["exp_avg_sq"], v_after)
    t_before = float(st["step"]) - 1
    vhat0 = (v_after - 0.001 * g[1].square()) / 0.999 / (1 - 0.999 ** t_before)
    assert torch.allclose(u[1], g[1] / (vhat0.sqrt() + 1e-8) * 0.01)


def test_update_map_clips_as_the_step():
    # the clip binds (|g| >> max norm): the map equals clip_grad_norm_ then a real MuonAdamW step, on both sets
    from optim.muon import MuonAdamW
    torch.manual_seed(6)
    W = torch.nn.Parameter(torch.randn(6, 4, dtype=torch.float64))
    bvec = torch.nn.Parameter(torch.randn(4, dtype=torch.float64))
    opt = MuonAdamW([W, bvec], lr=0.01, momentum=0.0, nesterov=False, muon_weight_decay=0.0, weight_decay=0.0,
                    mup=False, muon_filter=lambda p: p.ndim == 2)
    exp = SimpleNamespace(optimizer=opt, model=torch.nn.Module(),
                          cfg=OmegaConf.create({"training": {"clip_grad_norm": 0.5, "clip_grad_value": None}}))
    sets = {"muon": [W], "adamw": [bvec]}
    loss = lambda: 30 * (W.sum(1).square().sum() + (bvec * torch.arange(1.0, 5.0, dtype=torch.float64)).square().sum())
    for _ in range(3):                                            # a few clipped steps: AdamW's moments from clipped gradients
        opt.zero_grad(); loss().backward(); torch.nn.utils.clip_grad_norm_([W, bvec], 0.5); opt.step()
    opt.zero_grad(); loss().backward()
    g = [W.grad.clone(), bvec.grad.clone()]
    assert float(torch.cat([x.reshape(-1) for x in g]).norm()) > 10 * 0.5
    st = opt._adamw.state[bvec]
    v_before, t_before = st["exp_avg_sq"].clone(), float(st["step"])
    u = BaseExperiment._update_map(exp, g, sets)
    W0, b0 = W.detach().clone(), bvec.detach().clone()
    torch.nn.utils.clip_grad_norm_([W, bvec], 0.5); gc = bvec.grad.clone(); opt.step()
    assert torch.allclose(W0 - W.detach(), u[0], rtol=1e-4, atol=1e-7)
    vhat = v_before.double() / (1 - 0.999 ** t_before)
    assert torch.allclose(u[1], gc / (vhat.sqrt() + 1e-8) * 0.01)


def test_update_terms_known_answer():
    # per-example gradient G + xi, xi ~ N(0, diag sigma^2): a mean over b examples is G + N(0, sigma^2 / b). The
    # disjoint complement is an unbiased G, so A(b) = sum_i G_i E[u_i(g_b)]: for u = g it is |G|^2 at every b; for
    # u = sign(g) it is sum_i |G_i| erf(|G_i| sqrt(b) / (sigma_i sqrt 2)).
    import math
    torch.manual_seed(4)
    D, b, K, reps = 200, 16, 8, 300
    G = torch.cat([torch.full((5,), 1.0), torch.full((D - 5,), 0.02)]).double()   # |G| on few coordinates
    sig = torch.full((D,), 1.0, dtype=torch.float64)
    w = torch.nn.Parameter(torch.zeros(D, dtype=torch.float64))
    sets = {"all": [w]}
    for name, umap in (("identity", lambda g, s: g), ("sign", lambda g, s: [x.sign() for x in g])):
        exp = SimpleNamespace(_update_map=umap)
        acc = {}
        for _ in range(reps):
            per = [[G + sig * torch.randn(D, dtype=torch.float64) / math.sqrt(b)] for _ in range(K)]
            for kname, v in BaseExperiment._noise_update(exp, per, sets, b).items():
                acc[kname] = acc.get(kname, 0.0) + v / reps
        for m in (1, 2, 4):
            a = acc[f"all_A_b{m * b}"]
            if name == "identity":
                assert abs(a / float(G.square().sum()) - 1) < 0.05, (m, a)
            else:
                exact = float((G.abs() * torch.erf(G.abs() * math.sqrt(m * b) / (sig * math.sqrt(2)))).sum())
                assert abs(a / exact - 1) < 0.05, (m, a, exact)


def test_step_loss_decrease_known_answer():
    # chunk loss L_c(w) = 1/2 sum h (w - w*)^2 + <xi_c, w>, xi_c ~ N(0, sigma^2 / b): gradient G + xi_c with
    # G = h (w - w*). With the SGD map u = eta g and disjoint chunks for the read, E dL(b) =
    # eta |G|^2 - eta^2 / 2 (sum h G^2 + sum h sigma^2 / b); the weights come back exactly
    import math
    torch.manual_seed(5)
    D, b, K, reps, eta = 50, 8, 8, 400, 0.3
    h = torch.linspace(0.2, 2.0, D, dtype=torch.float64)
    w_star = torch.randn(D, dtype=torch.float64)
    w = torch.nn.Parameter(w_star + torch.randn(D, dtype=torch.float64))
    w0 = w.detach().clone()
    sig = 1.5
    sets = {"all": [w]}
    G = h * (w0 - w_star)
    exp = SimpleNamespace(_update_map=lambda g, s: [eta * x for x in g])
    acc = {}
    for _ in range(reps):
        xi = [sig * torch.randn(D, dtype=torch.float64) / math.sqrt(b) for _ in range(K)]
        L = lambda: [float(0.5 * (h * (w.detach() - w_star).square()).sum() + (x * w.detach()).sum()) for x in xi]
        rec = BaseExperiment._noise_update(exp, [[G + x] for x in xi], sets, b, dl=(L(), L), scales=(0.25, 1.0))
        for kname, v in rec.items():
            acc[kname] = acc.get(kname, 0.0) + v / reps
    assert torch.equal(w.detach(), w0)
    for m in (1, 2, 4):
        for s, key in ((1.0, f"dL_b{m * b}"), (0.25, f"dL_s0.25_b{m * b}")):   # the scaled step: eta -> s eta
            e = s * eta
            exact = float(e * G.square().sum() - e ** 2 / 2 * ((h * G.square()).sum() + (h * sig ** 2).sum() / (m * b)))
            assert abs(acc[key] / exact - 1) < 0.05, (m, s, acc[key], exact)


def test_dl_losses_share_the_rng_state():
    # a forward with random parts (lloca's frame noise) returns the same chunk losses at the same weights in one read,
    # and leaves the global RNG where it was
    exp = SimpleNamespace(model=None, _reg_term=lambda m: torch.tensor(0.0),
                          _noise_chunk_losses=lambda data, k: [torch.randn(()) for _ in range(k)])
    torch.manual_seed(0)
    before = torch.random.get_rng_state()
    a = BaseExperiment._noise_dl_losses(exp, None, 4, seed=17)
    b = BaseExperiment._noise_dl_losses(exp, None, 4, seed=17)
    assert a == b and len(set(a)) == 4
    assert torch.equal(torch.random.get_rng_state(), before)
