#!/usr/bin/env python
"""The per-process context vector (model.context_dim, D105; wrappers.AmplitudeLLoCaWrapper.setup_context) and the
K-shot tool's batch helpers (tools/kshot_context.py). CPU, a stub backbone that returns a fixed nonlinear function of
the particle features, so the wrapper's own logic is what is tested:
  (a) a zero table gives exactly the output without a context;
  (b) a process's row changes only its own events; a process with no row (-1) gets the zero vector;
  (c) context_dropout = 1 in train mode zeroes every event's context, eval mode never drops;
  (d) context_override replaces every event's vector and gradients reach it (and only it) with the rest frozen;
  (e) the rows follow names: a probe's run with another process order reads the parent's table by name;
  (f) kshot_context.subset / cat_batches rebuild ptr and the event slices exactly.
Run: python tests/test_context_vector.py   (needs torch and lgatr, i.e. a site's env)
"""
import os, sys

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from wrappers import AmplitudeLLoCaWrapper  # noqa: E402


class StubNet(nn.Module):
    """per particle: a fixed random projection of tanh(features), one output channel"""

    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, 1, generator=torch.Generator().manual_seed(1)))

    def forward(self, fourmomenta, particle_type, mean, std, ptr=None, seq_lens=None, pair_ctx=None, **kw):
        return torch.tanh(particle_type) @ self.w + 0.0 * fourmomenta[:, :1]


def make(names, own, dropout=0.0):
    torch.manual_seed(0)
    w = AmplitudeLLoCaWrapper(StubNet(16 + 2), token_size=0, d_particle_hidden=16)
    w.setup_particle_features(use_pids=False, property_matrix=torch.randn(5, 4).numpy())
    w.setup_context(names, 16, dropout, own)
    return w


def batch():
    ptr = torch.tensor([0, 3, 7, 10])                     # three events: 3, 4, 3 particles
    mom = torch.randn(10, 4)
    tok = torch.randint(1, 5, (10,))
    orders = torch.randn(3, 2)
    pids = torch.tensor([0, 1, 0])
    return mom, tok, orders, ptr, pids


def run(w, b, pids=None):
    mom, tok, orders, ptr, p = b
    return w(mom, tok, 0.0, 1.0, ptr, order_labels=orders, process_ids=p if pids is None else pids).reshape(-1)


def test_zero_table_is_identity():
    w, b = make(["a", "b"], ["a", "b"]), batch()
    ref = run(w, b)
    ctx, w.context = w.context, None
    assert torch.allclose(run(w, b), ref), "a zero table must leave the output unchanged"
    w.context = ctx


def test_rows_and_missing():
    w, b = make(["a", "b"], ["a", "c"]).eval(), batch()       # own process 1 ("c") has no row
    ref = run(w, b)
    with torch.no_grad():
        w.context[0] += 1.0                                       # row of "a" (own process 0: events 0 and 2)
    out = run(w, b)
    assert not torch.allclose(out[[0, 2]], ref[[0, 2]]), "process a's events must change"
    assert torch.allclose(out[1], ref[1]), "the event of the process with no row must not change"
    with torch.no_grad():
        w.context[1] += 5.0                                       # row "b" is no own process here
    assert torch.allclose(run(w, b)[1], ref[1])


def test_dropout():
    w, b = make(["a", "b"], ["a", "b"], dropout=1.0), batch()
    with torch.no_grad():
        w.context += 1.0
    w.train()
    ctx, w.context = w.context, None
    ref = run(w, b)
    w.context = ctx
    assert torch.allclose(run(w, b), ref), "dropout 1 in training must zero every context"
    w.eval()
    assert not torch.allclose(run(w, b), ref), "evaluation must not drop the context"


def test_override_and_grad():
    w, b = make(["a", "b"], ["a", "b"]).eval(), batch()
    for p in w.parameters():
        p.requires_grad_(False)
    ref = run(w, b)
    w.context_override = nn.Parameter(torch.zeros(16))
    assert torch.allclose(run(w, b), ref), "a zero override is the zero-shot model (the table is zero)"
    loss = run(w, b).sum()
    loss.backward()
    assert w.context_override.grad is not None and w.context_override.grad.abs().sum() > 0
    assert all(p.grad is None for n, p in w.named_parameters() if n != "context_override")
    with torch.no_grad():
        w.context_override += 1.0
    out = run(w, b)
    assert not torch.allclose(out, ref)
    # the same vector on every event, whatever its process id
    assert torch.allclose(run(w, b, pids=torch.tensor([1, 1, 1])), out)
    w.context_override = None


def test_rows_by_name():
    parent = make(["a", "b", "c"], ["a", "b", "c"])
    with torch.no_grad():
        parent.context.copy_(torch.randn(3, 16))
    probe = make(["a", "b", "c"], ["c"])                       # a probe run of one process, "c"
    probe.load_state_dict(parent.state_dict())
    assert probe._ctx_row.tolist() == [2]
    probe.eval(); parent.eval()
    b = batch()
    out_probe = run(probe, b, pids=torch.zeros(3, dtype=torch.long))
    out_parent = run(parent, b, pids=torch.full((3,), 2, dtype=torch.long))
    assert torch.allclose(out_probe, out_parent), "the probe must read the parent's row by name"


def test_subset_and_cat():
    from kshot_context import cat_batches, subset
    b1 = (torch.arange(20.).reshape(5, 4), torch.tensor([[1.], [2.]]), torch.arange(5), torch.zeros(2, 3),
          torch.tensor([0, 2, 5]), torch.tensor([0, 0]))
    b2 = (torch.arange(12.).reshape(3, 4) + 100, torch.tensor([[3.]]), torch.arange(3) + 10, torch.ones(1, 3),
          torch.tensor([0, 3]), torch.tensor([0]))
    c = cat_batches([b1, b2])
    assert c[4].tolist() == [0, 2, 5, 8] and c[1].reshape(-1).tolist() == [1., 2., 3.]
    s = subset(c, torch.tensor([0, 2]))
    assert s[4].tolist() == [0, 2, 5]
    assert torch.equal(s[0], torch.cat([c[0][0:2], c[0][5:8]]))
    assert s[1].reshape(-1).tolist() == [1., 3.] and s[2].tolist() == [0, 1, 10, 11, 12]


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print("ok", name)
    print("ALL OK")
