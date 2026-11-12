#!/usr/bin/env python
"""The three input representations for singular limits (D374, kit_benchmark input arms; config/model/lloca.yaml):
model.input_pair_logs, model.input_soft_eps, model.input_local_log_eps. For each flag: the μP LLoCa wrapper's output is
invariant under a random Lorentz transform of every momentum (float64), and the flag changes the output (it is wired).
Plus the pair logs' values on a known event and the soft rescale's fixed point x = 1.

Run:  .venv-cpu/bin/python tests/test_input_reps.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wrappers import AmplitudeLLoCaWrapper, _pair_log_inputs, _soft_rescale  # noqa: E402

torch.set_default_dtype(torch.float64)
LENS = (5, 5, 4)


def _events(seed, lens=LENS):
    """Events of `lens` particles: two back-to-back massless beams, then massless outgoing momenta, one of them soft."""
    rng = np.random.default_rng(seed)
    out = []
    for n in lens:
        p3 = rng.normal(size=(n - 2, 3))
        p3[-1] *= 1e-6                                    # a soft leg
        ev = [[1, 0, 0, 1], [1, 0, 0, -1]] + [[np.linalg.norm(v), *v] for v in p3]
        out.append(ev)
    return torch.tensor(np.concatenate(out)), torch.tensor(np.cumsum([0, *lens]))


def _lorentz(seed):
    rng = np.random.default_rng(seed)
    b = rng.normal(size=3)
    b *= 0.7 / np.linalg.norm(b)
    g = 1 / np.sqrt(1 - b @ b)
    B = np.eye(4)
    B[0, 0] = g
    B[0, 1:] = B[1:, 0] = g * b
    B[1:, 1:] += (g - 1) * np.outer(b, b) / (b @ b)
    R = np.eye(4)
    R[1:, 1:] = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    return torch.tensor(B @ R)


def _wrapper(**flags):
    from models.lloca import LLOCAMuPTransformer
    n_pl = flags.get("input_pair_logs", 0)
    ns = 16 + n_pl * (n_pl - 1) // 2
    torch.manual_seed(0)
    net = LLOCAMuPTransformer(num_scalars=ns, hidden_channels_mlp=16, num_layers_mlp=2, in_channels=ns + 4,
                              attn_reps="8x0n+2x1n", out_channels=1, num_blocks=2, num_heads=2)
    w = AmplitudeLLoCaWrapper(net, token_size=0, d_particle_hidden=16, **flags)
    w.setup_particle_features(use_pids=False, property_matrix=torch.randn(5, 4).numpy())
    return w.double().eval()


def _out(w, p, ptr):
    tok = torch.cat([torch.tensor([1, 1, 2, 3, 4, 2])[:n] for n in (ptr[1:] - ptr[:-1]).tolist()])
    with torch.no_grad():
        return w(p, tok, 0.0, 1.0, ptr).reshape(-1)


FLAGS = {"pair_logs": dict(input_pair_logs=5), "soft_rescale": dict(input_soft_eps=1e-8),
         "local_logs": dict(input_local_log_eps=1e-8),
         "both_6": dict(input_pair_logs=6, input_soft_eps=1e-8)}     # the uugg flags: 15 pair logs + soft rescale


def _check(name, lens=LENS):
    p, ptr = _events(1, lens)
    L = _lorentz(2)
    w = _wrapper(**FLAGS[name])
    a, b = _out(w, p, ptr), _out(w, p @ L.T, ptr)
    rel = float((a - b).abs().max() / a.abs().max())
    # float64 frame rounding (~1e-16) is amplified near z = 0 by up to 1/(eps asinh(1/eps)) ~ 5e6 under local_logs
    assert rel < 1e-7, f"{name}: not Lorentz invariant, rel {rel:.2e}"
    moved = float((a - _out(_wrapper(), p, ptr)).abs().max() / a.abs().max())
    assert moved > 1e-6, f"{name}: the flag does not change the output"
    print(f"{name}: invariant (rel {rel:.1e}), flag moves the output (rel {moved:.1e}): ok")


def test_pair_logs_equivariant():
    _check("pair_logs")


def test_soft_rescale_equivariant():
    _check("soft_rescale")


def test_local_logs_equivariant():
    _check("local_logs")


def test_both_flags_6particle_equivariant():
    _check("both_6", lens=(6, 5, 4))


def test_pair_log_values():
    p, ptr = _events(3)
    f = _pair_log_inputs(p, ptr, 5)
    assert f.shape == (14, 10)
    i, j, k = 2, 4, 0                                  # event 0: pair (2, 4) is column 7 of the i<j order
    col = [(a, c) for a in range(5) for c in range(a + 1, 5)].index((i, j))
    dot = p[i, 0] * p[j, 0] - (p[i, 1:] * p[j, 1:]).sum()
    assert torch.allclose(f[k, col], torch.log(2 * dot / 4.0) / 5)
    assert (f[10:, [3, 6, 8, 9]] == 0).all()           # the 4-particle event's pairs with leg 4 are 0
    assert (f[0] == f[4]).all()                        # broadcast over the event


def test_pair_threshold_log_beta():
    """input_pair_threshold: a t t~ pair at sqrt s gives log(beta^2)/5, beta^2 = 1 - 4 m^2/s; massless pairs unchanged."""
    m, sq = 173.0, 346.5
    E, pz = sq / 2, (sq * sq / 4 - m * m) ** 0.5
    p = torch.tensor([[E, 0, 0, E], [E, 0, 0, -E], [E, pz * 0.6, 0, pz * 0.8], [E, -pz * 0.6, 0, -pz * 0.8]],
                     dtype=torch.float64)
    ptr = torch.tensor([0, 4])
    f, g = _pair_log_inputs(p, ptr, 4, threshold=True), _pair_log_inputs(p, ptr, 4)
    beta2 = 1 - 4 * m * m / sq ** 2
    assert torch.allclose(f[0, 5], torch.log(torch.tensor(beta2, dtype=torch.float64)) / 5)   # pair (2, 3)
    assert torch.allclose(f[0, :5], g[0, :5])                                                  # pairs with a massless leg


def test_scalars_only_reads_no_momentum():
    """input_scalars_only: with the pair logs held fixed, two different event sets give the same output (no momentum
    reaches the backbone but through the pair logs), and the flag changes the output against the pair logs alone."""
    import wrappers
    p, ptr = _events(1)
    q, _ = _events(5)
    fixed = _pair_log_inputs(p, ptr, 5)
    w = _wrapper(input_pair_logs=5, input_scalars_only=True)
    orig = wrappers._pair_log_inputs
    wrappers._pair_log_inputs = lambda *a, **k: fixed
    try:
        a, b = _out(w, p, ptr), _out(w, q, ptr)
    finally:
        wrappers._pair_log_inputs = orig
    assert torch.equal(a, b), f"scalars_only reads the momenta: {float((a - b).abs().max()):.2e}"
    moved = float((a - _out(_wrapper(input_pair_logs=5), p, ptr)).abs().max() / a.abs().max())
    assert moved > 1e-6, "input_scalars_only does not change the output"


def test_soft_rescale_fixed_point():
    p, ptr = _events(4)
    q = _soft_rescale(p, ptr, 1e-8)
    assert torch.allclose(q[:2], p[:2])                # the beams have x = 1
    assert float(q[4, 0] / p[4, 0]) > 1e5              # the soft leg is lifted (x ~ 1e-6: log1p(100)/(1e-6 log 1e8))


if __name__ == "__main__":
    for t in [test_scalars_only_reads_no_momentum, test_pair_log_values, test_soft_rescale_fixed_point, test_pair_logs_equivariant,
              test_soft_rescale_equivariant, test_local_logs_equivariant, test_both_flags_6particle_equivariant]:
        t()
    print("all ok")
