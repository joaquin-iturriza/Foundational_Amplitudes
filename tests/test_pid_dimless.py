#!/usr/bin/env python
"""One-hot PIDs under data.dimensionless_inputs (wrappers.AmplitudeLLoCaWrapper.setup_dimless_pids). CPU, a stub
backbone that returns a fixed nonlinear function of the particle features, so the wrapper's own logic is tested:
  (a) the event's log10 sqrt(s_hat), the last order column, never reaches the backbone: changing it leaves the output;
  (b) the other order columns do reach it, broadcast to every particle after the one-hot token.
Run: python tests/test_pid_dimless.py   (needs torch and lgatr, i.e. a site's or the CPU env)
"""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import os, sys

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wrappers import AmplitudeLLoCaWrapper  # noqa: E402

TOK, NORD = 5, 2


class StubNet(nn.Module):
    """per particle: a fixed random projection of tanh(features), one output channel; asserts the feature width"""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.w = nn.Parameter(torch.randn(d, 1, generator=torch.Generator().manual_seed(1)))

    def forward(self, fourmomenta, particle_type, mean, std, ptr=None, seq_lens=None, pair_ctx=None, **kw):
        assert particle_type.shape[1] == self.d, (particle_type.shape, self.d)
        return torch.tanh(particle_type) @ self.w + 0.0 * fourmomenta[:, :1]


def make():
    w = AmplitudeLLoCaWrapper(StubNet(TOK + NORD), token_size=TOK, d_particle_hidden=16)
    w.setup_particle_features(use_pids=True)
    w.setup_dimless_pids()
    return w


def run(w, orders):
    torch.manual_seed(0)
    ptr = torch.tensor([0, 3, 7, 10])
    mom, tok = torch.randn(10, 4), torch.randint(1, TOK, (10,))
    return w(mom, tok, 0.0, 1.0, ptr, order_labels=orders).reshape(-1)


def test_lrs_column_dropped():
    w = make()
    o = torch.randn(3, NORD + 1)
    o2 = o.clone(); o2[:, -1] += 3.0
    assert torch.equal(run(w, o), run(w, o2))
    print("log10 sqrt(s_hat) column dropped: ok")


def test_order_columns_kept():
    w = make()
    o = torch.randn(3, NORD + 1)
    o2 = o.clone(); o2[:, 0] += 1.0
    assert not torch.allclose(run(w, o), run(w, o2))
    print("order columns kept: ok")


if __name__ == "__main__":
    test_lrs_column_dropped()
    test_order_columns_kept()
    print("all passed")
