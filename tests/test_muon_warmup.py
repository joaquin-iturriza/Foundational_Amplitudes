"""CPU check of MuonAdamW.muon_lr_mult (fine_tune.muon_warmup_steps, D303): at 0 the Muon matrices stay bit-identical
while AdamW's parameters move as without it; at 1 (the default) a step equals the unmodified optimizer's; at 0.25 the
Muon matrix moves by a quarter of its full step (same momentum).
    python tests/test_muon_warmup.py        (or pytest tests/test_muon_warmup.py)
"""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optim.muon import MuonAdamW  # noqa: E402


class _Inf:
    def ninf(self):
        return 2

    def width_mult(self):
        return 1.0


def _setup(seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(8, 8, generator=g) * 0.3); W.infshape = _Inf()
    b = torch.nn.Parameter(torch.randn(8, generator=g))
    opt = MuonAdamW([W, b], lr=1e-2)
    W.grad = torch.randn(8, 8, generator=g); b.grad = torch.randn(8, generator=g)
    return W, b, opt


def test_mult_zero_freezes_muon_only():
    W, b, opt = _setup()
    W0, b0 = W.detach().clone(), b.detach().clone()
    opt.muon_lr_mult = 0.0
    opt.step()
    assert torch.equal(W.detach(), W0)
    W2, b2, opt2 = _setup()
    opt2.step()
    assert torch.equal(b.detach(), b2.detach()) and not torch.equal(b.detach(), b0)


def test_mult_one_is_default():
    W, b, opt = _setup(); opt.muon_lr_mult = 1.0; opt.step()
    W2, b2, opt2 = _setup(); opt2.step()
    assert torch.equal(W.detach(), W2.detach()) and torch.equal(b.detach(), b2.detach())


def test_mult_scales_muon_step():
    W, _, opt = _setup(); W0 = W.detach().clone(); opt.muon_lr_mult = 0.25; opt.step()
    W2, _, opt2 = _setup(); opt2.step()
    # Muon's decoupled decay also scales with lr: compare the steps, decay included
    assert torch.allclose(W.detach() - W0, 0.25 * (W2.detach() - W0), rtol=1e-3, atol=1e-6)


if __name__ == "__main__":
    for f in (test_mult_zero_freezes_muon_only, test_mult_one_is_default, test_mult_scales_muon_step):
        f(); print("ok", f.__name__)
