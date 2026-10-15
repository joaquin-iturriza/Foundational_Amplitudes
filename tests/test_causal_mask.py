#!/usr/bin/env python
"""Causal (autoregressive) attention for the multiplicity extrapolation (paper draft, Sec. "Extrapolating to more
legs"): model.net.causal_mask, causal_frames, data.particle_order, data.affine_multiplicity_std.

Part 1 (numpy, any machine): the particle order (multiplicity_extrap.order_identical_particles) and the affine
multiplicity standardization fit.
Part 2 (torch + lloca + mup; GPU node via scripts/job_test_causal.sh, the laptop has no torch):
  - the mask: lower-triangular inside each event, block-diagonal across events (dense and xformers forms agree);
  - the frames-net edges: particle i reads 1..max(i, K) only;
  - causal_mask=False is the model built without the new arguments;
  - the leakage test: per-particle outputs of a prefix of length >= K do not change when the particles after it
    are removed, through the attention AND the frames net; negative controls: causal_frames="full" and
    causal_mask=False do leak;
  - with xformers on CUDA: the same leakage test on the production attention path.

Run:  python tests/test_causal_mask.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from multiplicity_extrap import order_identical_particles, fit_affine_multiplicity  # noqa: E402


def _mdot(p, q):
    return p[..., 0] * q[..., 0] - (p[..., 1:] * q[..., 1:]).sum(-1)


def _events(rng, n_ev, P, mass=None):
    """Random momenta (n_ev, P, 4) with positive energy (masses in [0, 1])."""
    p3 = rng.normal(size=(n_ev, P, 3))
    m = rng.uniform(0, 1, size=(n_ev, P)) if mass is None else np.full((n_ev, P), mass)
    E = np.sqrt((p3 ** 2).sum(-1) + m ** 2)
    return np.concatenate([E[..., None], p3], -1)


# ----------------------------------------------------------------------------------------------- part 1 (numpy)
def test_order_pdotP():
    rng = np.random.default_rng(1)
    pdg = [2, -1, 24, 21, 21, 21]
    mom = _events(rng, 500, 6)
    out = order_identical_particles(mom, np.tile(pdg, (500, 1)), "pdotP")
    assert out.shape == mom.shape
    # beams and the W keep their slots
    assert np.array_equal(out[:, :3], mom[:, :3])
    # the gluons are the same multiset per event, sorted by p.P descending, P = summed outgoing momentum
    P = mom[:, 2:].sum(1)
    key = _mdot(out[:, 3:], P[:, None, :])
    assert (np.diff(key, axis=1) <= 1e-12).all()
    assert np.allclose(np.sort(_mdot(mom[:, 3:], P[:, None]), 1), np.sort(key, 1))
    assert np.allclose(np.sort(out[:, 3:, 0], 1), np.sort(mom[:, 3:, 0], 1))
    # P (and so the order) is unchanged by the reordering; idempotent; invariant under a permutation of the gluons
    assert np.allclose(out[:, 2:].sum(1), P)
    assert np.array_equal(order_identical_particles(out, pdg, "pdotP"), out)
    perm = mom.copy()
    perm[:, 3:] = mom[:, [5, 3, 4]]
    assert np.array_equal(order_identical_particles(perm, pdg, "pdotP"), out)
    # p.P is Lorentz invariant: ordering a boosted event gives the boosted ordered event
    b = 0.6
    g = 1 / np.sqrt(1 - b ** 2)
    L = np.array([[g, 0, 0, g * b], [0, 1, 0, 0], [0, 0, 1, 0], [g * b, 0, 0, g]])
    assert np.allclose(order_identical_particles(mom @ L.T, pdg, "pdotP"), out @ L.T)
    # flat rows (the .npy layout) give the same
    flat = order_identical_particles(mom.reshape(500, 24), pdg, "pdotP")
    assert flat.shape == (500, 24) and np.array_equal(flat.reshape(500, 6, 4), out)
    print("order pdotP (W+3g layout): ok")


def test_order_groups():
    rng = np.random.default_rng(2)
    # u u -> u u: the beams never move although they share the outgoing PDG id; the two outgoing u are sorted
    mom = _events(rng, 200, 4)
    out = order_identical_particles(mom, [2, 2, 2, 2], "pdotP")
    assert np.array_equal(out[:, :2], mom[:, :2])
    P = mom[:, 2:].sum(1)
    assert (_mdot(out[:, 2], P) >= _mdot(out[:, 3], P) - 1e-12).all()
    # no identical outgoing particles, or mode none: untouched
    mom = _events(rng, 50, 5)
    assert order_identical_particles(mom, [2, -2, 23, 21, 1], "pdotP") is mom
    assert order_identical_particles(mom, [2, -2, 23, 21, 21], "none") is mom
    # two groups (photons and gluons interleaved): each permuted within its own slots
    mom = _events(rng, 300, 6)
    pdg = [11, -11, 22, 21, 22, 21]
    out = order_identical_particles(mom, pdg, "pdotP")
    P = mom[:, 2:].sum(1)
    for grp in ([2, 4], [3, 5]):
        k = _mdot(out[:, grp], P[:, None])
        assert (k[:, 0] >= k[:, 1] - 1e-12).all()
        assert np.allclose(np.sort(out[:, grp, 0], 1), np.sort(mom[:, grp, 0], 1))
    # shuffle: same slots, same multisets, beams fixed, actually permutes
    sh = order_identical_particles(mom, pdg, "shuffle", rng=np.random.default_rng(0))
    assert np.array_equal(sh[:, :2], mom[:, :2])
    for grp in ([2, 4], [3, 5]):
        assert np.allclose(np.sort(sh[:, grp, 0], 1), np.sort(mom[:, grp, 0], 1))
    assert not np.array_equal(sh, mom)
    print("order groups / beams / shuffle: ok")


def test_affine_fit():
    rng = np.random.default_rng(3)
    a, b = 3.0, -2.5
    N = np.array([4, 4, 5, 6, 4])
    n = np.array([1000, 3000, 2000, 1500, 500])
    sd = np.array([1.0, 0.5, 1.5, 2.0, 1.0])
    # means exactly on the line: the fit recovers it and sigma is the pooled std
    a2, b2, s2 = fit_affine_multiplicity(n, N, a + b * N, sd)
    assert np.isclose(a2, a) and np.isclose(b2, b)
    assert np.isclose(s2, np.sqrt((n * sd ** 2).sum() / n.sum()))
    # off the line: equal to ordinary least squares of y on N over all events, sigma the rms residual
    mu = a + b * N + rng.normal(size=5)
    ys, xs, ms, ss = [], [], [], []
    for k in range(5):
        y = rng.normal(size=n[k])
        y = (y - y.mean()) / y.std() * sd[k] + mu[k]
        ys.append(y); xs.append(np.full(n[k], N[k])); ms.append(y.mean()); ss.append(y.std())
    y, x = np.concatenate(ys), np.concatenate(xs)
    B, A = np.polyfit(x, y, 1)
    a3, b3, s3 = fit_affine_multiplicity(n, N, ms, ss)
    assert np.isclose(a3, A) and np.isclose(b3, B)
    assert np.isclose(s3, np.sqrt(np.mean((y - A - B * x) ** 2)))
    try:
        fit_affine_multiplicity([1, 1], [4, 4], [0, 1], [1, 1])
    except AssertionError:
        pass
    else:
        raise RuntimeError("a single multiplicity must be refused")
    print("affine multiplicity fit: ok")


# ----------------------------------------------------------------------------------------------- part 2 (torch)
def _torch():
    try:
        import torch  # noqa: F401
        from models.lloca import LLOCAMuPTransformer  # noqa: F401  (needs lloca, mup, torch_geometric)
        return True
    except Exception as exc:  # the laptop has none of them
        print(f"torch part: skipped ({type(exc).__name__}: {exc})")
        return False


def _ptr(lens):
    import torch
    return torch.tensor(np.concatenate([[0], np.cumsum(lens)]), dtype=torch.long)


def test_dense_mask():
    import torch
    from models.attention_lloca_mup import dense_block_causal_mask
    lens = [4, 6, 5, 7]
    m = dense_block_causal_mask(_ptr(lens))
    ref = torch.block_diag(*[torch.ones(k, k, dtype=torch.bool).tril() for k in lens])
    assert torch.equal(m, ref)
    print("dense causal mask: lower-triangular per event, block-diagonal across: ok")


def test_xformers_mask():
    import torch
    import models.attention_lloca_mup as A
    if not (A._XFORMERS_AVAILABLE and torch.cuda.is_available()):
        print("xformers causal mask: skipped (needs xformers and CUDA)")
        return
    lens = [4, 6, 5, 7]
    ptr = _ptr(lens)
    bias = A.build_block_diagonal_bias(ptr, causal=True)
    N = sum(lens)
    dense = bias.materialize((N, N), dtype=torch.float32, device="cuda")
    assert torch.equal(torch.isfinite(dense).cpu(), A.dense_block_causal_mask(ptr))
    # attention through xformers equals the dense masked attention
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 2, N, 8, generator=g).cuda() for _ in range(3))
    x = A.scaled_dot_product_attention(q, k, v, attn_bias=bias)
    d = A.scaled_dot_product_attention(q, k, v, attn_bias=A.dense_block_causal_mask(ptr, device="cuda"))
    assert torch.allclose(x, d, atol=1e-5), (x - d).abs().max()
    print("xformers BlockDiagonalCausalMask == dense mask (pattern and attention): ok")


def test_causal_edges():
    import torch
    from models.equimlp import get_edge_index_from_ptr, restrict_edges_causal
    lens, K = [4, 6, 7], 4
    ptr = _ptr(lens)
    N = int(ptr[-1])
    ei, eptr = restrict_edges_causal(get_edge_index_from_ptr(ptr), ptr, N, K)
    got = set(map(tuple, ei.t().tolist()))
    want = set()
    for e, n in enumerate(lens):
        o = int(ptr[e])
        for i in range(n):
            for j in range(n):
                if i != j and j <= max(i, K - 1):
                    want.add((o + i, o + j))
    assert got == want
    assert (torch.diff(ei[0]) >= 0).all()                     # grouped by the receiving node
    assert torch.equal(torch.diff(eptr), torch.bincount(ei[0], minlength=N))
    assert (torch.diff(eptr) > 0).all()                       # every node has an edge (a frame)
    print("frames-net causal edges: i reads 1..max(i, K): ok")


def _model(causal_mask, causal_frames="prefix", K=4, seed=0, new_args=True):
    import torch
    from models.lloca import LLOCAMuPTransformer
    torch.manual_seed(seed)
    kw = dict(causal_mask=causal_mask, causal_frames=causal_frames, causal_frames_min=K) if new_args else {}
    return LLOCAMuPTransformer(num_scalars=4, hidden_channels_mlp=16, num_layers_mlp=2, in_channels=8,
                               attn_reps="8x0n+2x1n", out_channels=1, num_blocks=2, num_heads=2, **kw)


def _batch(seed, lens):
    import torch
    rng = np.random.default_rng(seed)
    mom = [torch.tensor(_events(rng, 1, n)[0], dtype=torch.float32) for n in lens]
    sca = [torch.tensor(rng.normal(size=(n, 4)), dtype=torch.float32) for n in lens]
    return mom, sca


def _run(net, ms, ss, device):
    import torch
    ptr = _ptr([len(m) for m in ms]).to(device)
    with torch.no_grad():
        return net(torch.cat(ms).to(device), torch.cat(ss).to(device), 0.0, 1.0, ptr).float().cpu(), ptr.cpu()


def _leak(net, device, K=4, lens=(5, 7, 6, 4)):
    """max over prefixes L >= K of |out_prefix(full event) - out(event cut to L)| / max|out|."""
    mom, sca = _batch(5, lens)
    full, ptr = _run(net, mom, sca, device)
    scale = full.abs().max().item()
    worst = 0.0
    for L in range(K, max(lens)):
        cut, cptr = _run(net, [m[:L] for m in mom], [s[:L] for s in sca], device)
        for e in range(len(lens)):
            k = min(L, lens[e])
            a = full[int(ptr[e]):int(ptr[e]) + k]
            b = cut[int(cptr[e]):int(cptr[e]) + k]
            worst = max(worst, (a - b).abs().max().item() / scale)
    return worst


def _frames_leak(net, device, K=4, lens=(5, 7, 6), dtype=None):
    """The frames of the first K particles: full event against the event cut to K."""
    import torch
    mom, sca = _batch(6, lens)
    if dtype is not None:
        mom, sca = [m.to(dtype) for m in mom], [s.to(dtype) for s in sca]
    kw = {"causal_min_prefix": K} if net.causal_mask and net.causal_frames == "prefix" else {}

    def frames(ms, ss):
        ptr = _ptr([len(m) for m in ms]).to(device)
        with torch.no_grad():
            f = net.framesnet(torch.cat(ms).to(device), scalars=torch.cat(ss).to(device), ptr=ptr,
                              return_tracker=False, **kw)
        return f.matrices.cpu(), ptr.cpu()

    full, ptr = frames(mom, sca)
    cut, cptr = frames([m[:K] for m in mom], [s[:K] for s in sca])
    d = max((full[int(ptr[e]):int(ptr[e]) + K] - cut[int(cptr[e]):int(cptr[e]) + K]).abs().max().item()
            for e in range(len(lens)))
    return d, bool(torch.isfinite(full).all())


def test_frames_leak_fp64():
    """The frames net alone, in float64 (the backbone casts to float32, the frames net does not): the frames of the
    first K particles are exactly those of the event cut to K (float32 rounding through the polar decomposition
    moves them by ~1e-4, hence float64 here)."""
    import torch
    torch.set_default_dtype(torch.float64)
    try:
        net = _model(True).eval()
        d, finite = _frames_leak(net, "cpu", dtype=torch.float64)
        assert finite and d < 1e-10, f"frames of the first K particles move with later particles: {d}"
        nf = _model(True, causal_frames="full").eval()
        df, _ = _frames_leak(nf, "cpu", dtype=torch.float64)
        assert df > 1e-3, f"negative control: full-event frames should move, got {df}"
    finally:
        torch.set_default_dtype(torch.float32)
    print(f"frames (float64): prefix frames move by {d:.1e} when later particles are removed; full-event frames "
          f"{df:.1e}: ok")


def _leakage_suite(device, tol):
    """Outputs, float32: the causal model's prefix outputs move only by rounding; both controls move by O(1)."""
    net = _model(True).to(device).eval()
    _leak(net, device)                          # the first call fixes the frames net's edge standardization
    w = _leak(net, device)
    assert w < tol, f"causal (prefix frames): a prefix output moves when later particles are removed: {w}"
    nf = _model(True, causal_frames="full").to(device).eval()
    _leak(nf, device)
    wf = _leak(nf, device)
    assert wf > 1e-2, f"negative control: full-event frames should leak, got {wf}"
    nc = _model(False).to(device).eval()
    _leak(nc, device)
    wc = _leak(nc, device)
    assert wc > 1e-2, f"negative control: no mask should leak, got {wc}"
    print(f"leakage on {device} (relative to max|output|): causal + prefix frames {w:.1e}; controls: full-event "
          f"frames {wf:.1e}, no mask {wc:.1e}: ok")


class _dense_attention:
    """xformers has no CPU kernel: route the CPU tests through the dense masked path."""
    def __enter__(self):
        import models.attention_lloca_mup as A
        self.A, self.saved = A, A._XFORMERS_AVAILABLE
        A._XFORMERS_AVAILABLE = False

    def __exit__(self, *exc):
        self.A._XFORMERS_AVAILABLE = self.saved


def test_off_is_unchanged():
    """causal_mask=False is the model built without the new arguments (same seed, same output, bit for bit)."""
    import torch
    torch.set_default_dtype(torch.float32)
    old = _model(False, new_args=False).eval()
    new = _model(False).eval()
    mom, sca = _batch(7, (4, 5, 6))
    with _dense_attention():
        assert torch.equal(_run(old, mom, sca, "cpu")[0], _run(new, mom, sca, "cpu")[0])
    if torch.cuda.is_available():
        old, new = old.cuda(), new.cuda()
        assert torch.equal(_run(old, mom, sca, "cuda")[0], _run(new, mom, sca, "cuda")[0])
    print("causal_mask=False: identical to the model built without the new arguments: ok")


def test_leakage_cpu_dense():
    import torch
    torch.set_default_dtype(torch.float32)
    with _dense_attention():
        _leakage_suite("cpu", tol=1e-4)


def test_leakage_cuda_xformers():
    import torch
    import models.attention_lloca_mup as A
    if not (A._XFORMERS_AVAILABLE and torch.cuda.is_available()):
        print("leakage through xformers: skipped (needs xformers and CUDA)")
        return
    torch.set_default_dtype(torch.float32)
    _leakage_suite("cuda", tol=1e-3)


if __name__ == "__main__":
    test_order_pdotP()
    test_order_groups()
    test_affine_fit()
    if _torch():
        test_dense_mask()
        test_xformers_mask()
        test_causal_edges()
        test_off_is_unchanged()
        test_frames_leak_fp64()
        test_leakage_cpu_dense()
        test_leakage_cuda_xformers()
    print("all passed")
