"""CPU check of BaseExperiment._load_parent_optimizer_state (fine_tune.load_optimizer_state, plan warm_start step 13):
a parent trained a few steps and saved as _save_model does (model + optimizer state, gzipped), then a fresh MuonAdamW
on a copy of the model at another lr gets the parent's per-parameter state (Muon momentum, AdamW moments and step)
while keeping its own lr; its next step equals the parent's own next step rescaled by the lr; a mismatched layout raises.
    python tests/test_load_optimizer_state.py        (or pytest tests/test_load_optimizer_state.py)
"""
import gzip
import logging
import os
import sys
import tempfile
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import base_experiment  # noqa: E402
from base_experiment import BaseExperiment  # noqa: E402
from optim.muon import MuonAdamW  # noqa: E402


class _Inf:
    def ninf(self):
        return 2

    def width_mult(self):
        return 1.0


class _Stub:
    _load_parent_optimizer_state = BaseExperiment._load_parent_optimizer_state

    def __init__(self, lr, extra=False):
        g = torch.Generator().manual_seed(0)
        self.model = torch.nn.Module()
        self.model.W = torch.nn.Parameter(torch.randn(8, 8, generator=g) * 0.3)
        self.model.W.infshape = _Inf()
        self.model.b = torch.nn.Parameter(torch.randn(8, generator=g) * 0.1)
        ps = [self.model.W, self.model.b]
        if extra:
            self.model.c = torch.nn.Parameter(torch.zeros(8))
            ps.append(self.model.c)
        self.optimizer = MuonAdamW([{"params": ps, "lr": lr}], lr=lr, muon_weight_decay=0.0, weight_decay=0.0)
        self.device = "cpu"


def _step(s, seed):
    g = torch.Generator().manual_seed(seed)
    x, y = torch.randn(32, 8, generator=g), torch.randn(32, 8, generator=g)
    s.optimizer.zero_grad()
    ((x @ s.model.W.T + s.model.b - y) ** 2).mean().backward()
    s.optimizer.step()


def _parent_ckpt(tmp):
    p = _Stub(lr=1e-2)
    for i in range(5):
        _step(p, i)
    path = os.path.join(tmp, "parent.pt.gz")
    with gzip.open(path, "wb") as f:
        torch.save({"model": p.model.state_dict(), "optimizer": p.optimizer.state_dict()}, f)
    return p, path


def test_state_loaded_lr_kept():
    base_experiment.LOGGER.setLevel(logging.WARNING)
    with tempfile.TemporaryDirectory() as tmp:
        p, path = _parent_ckpt(tmp)
        r = _Stub(lr=2e-3)
        r.model.load_state_dict(p.model.state_dict())
        r._load_parent_optimizer_state(path)
        assert [g["lr"] for g in r.optimizer.param_groups] == [2e-3] * len(r.optimizer.param_groups)
        assert torch.equal(r.optimizer.state[r.model.W]["momentum_buffer"], p.optimizer.state[p.model.W]["momentum_buffer"])
        st, ps = r.optimizer._adamw.state[r.model.b], p.optimizer._adamw.state[p.model.b]
        assert int(st["step"]) == 5 and torch.equal(st["exp_avg_sq"], ps["exp_avg_sq"])
        assert r.opt_state_loaded == {"entries": 1, "adamw_entries": 1}
        # the next step continues the parent's: same direction, the size of this run's lr
        W0 = r.model.W.detach().clone(); b0 = r.model.b.detach().clone()
        Wp0 = p.model.W.detach().clone(); bp0 = p.model.b.detach().clone()
        _step(r, 99); _step(p, 99)
        assert torch.allclose((r.model.W - W0) * 5, p.model.W - Wp0, atol=1e-6)
        assert torch.allclose((r.model.b - b0) * 5, p.model.b - bp0, atol=1e-6)


def test_layout_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        _, path = _parent_ckpt(tmp)
        r = _Stub(lr=2e-3, extra=True)
        try:
            r._load_parent_optimizer_state(path)
        except RuntimeError as e:
            assert "groups differ" in str(e) or "shape" in str(e)
        else:
            raise AssertionError("a different parameter layout must raise")


if __name__ == "__main__":
    test_state_loaded_lr_kept(); test_layout_mismatch_raises(); print("ok")
