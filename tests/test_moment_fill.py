"""CPU check of BaseExperiment._fill_optimizer_moments (fine_tune.moment_fill_steps, D303): n optimizer steps at lr 0
leave every weight bit-identical, fill AdamW's moments (step count n, exp_avg_sq > 0) and Muon's momentum buffer, skip a
non-finite step, restore each group's lr, and the first real step after a fill is gradient-sized for AdamW (smaller
than the fresh optimizer's lr-sized first step on a noisy gradient).
    python tests/test_moment_fill.py        (or pytest tests/test_moment_fill.py)
"""
import logging
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import base_experiment  # noqa: E402
from base_experiment import BaseExperiment  # noqa: E402
from optim.muon import MuonAdamW  # noqa: E402


class _Inf:
    def __init__(self, ninf, m):
        self._n, self._m = ninf, m

    def ninf(self):
        return self._n

    def width_mult(self):
        return self._m


class _Stub:
    """the parts of an experiment _fill_optimizer_moments touches: a model with a Muon matrix and an AdamW vector,
    the optimizer, cfg.training.clip_grad_norm, _batch_loss on (x, y) batches"""
    _fill_optimizer_moments = BaseExperiment._fill_optimizer_moments

    def __init__(self, seed=0, nan_at=None):
        g = torch.Generator().manual_seed(seed)
        self.model = torch.nn.Module()
        self.model.W = torch.nn.Parameter(torch.randn(8, 8, generator=g) * 0.3)
        self.model.W.infshape = _Inf(2, 1.0)
        self.model.b = torch.nn.Parameter(torch.randn(8, generator=g) * 0.1)
        self.optimizer = MuonAdamW([{"params": [self.model.W, self.model.b], "lr": 1e-2}], lr=1e-2,
                                   muon_weight_decay=0.1, weight_decay=0.01)
        self.cfg = SimpleNamespace(training=SimpleNamespace(clip_grad_norm=5.0))
        self.calls, self.nan_at = 0, nan_at

    def _batch_loss(self, data):
        x, y = data
        self.calls += 1
        out = x @ self.model.W.T + self.model.b
        loss = ((out - y) ** 2).mean()
        if self.nan_at is not None and self.calls == self.nan_at:
            loss = loss * float("nan")
        return loss, loss, None


def _batches(seed=1):
    g = torch.Generator().manual_seed(seed)
    while True:
        x = torch.randn(32, 8, generator=g)
        yield x, torch.randn(32, 8, generator=g)


def test_fill_keeps_weights_and_fills_state():
    base_experiment.LOGGER.setLevel(logging.WARNING)
    s = _Stub()
    W0, b0 = s.model.W.detach().clone(), s.model.b.detach().clone()
    lrs = [g["lr"] for g in s.optimizer.param_groups]
    s._fill_optimizer_moments(_batches(), 20)
    assert torch.equal(s.model.W, W0) and torch.equal(s.model.b, b0)
    assert [g["lr"] for g in s.optimizer.param_groups] == lrs
    assert s.moment_fill == {"asked": 20, "taken": 20, "max_weight_change": 0.0}
    assert s.optimizer.state[s.model.W]["momentum_buffer"].abs().sum() > 0
    st = s.optimizer._adamw.state[s.model.b]
    assert int(st["step"]) == 20 and (st["exp_avg_sq"] > 0).all()
    assert all(p.grad is None or not p.grad.abs().sum() for p in s.model.parameters())


def test_nonfinite_step_skipped():
    s = _Stub(nan_at=3)
    s._fill_optimizer_moments(_batches(), 10)
    assert s.moment_fill["taken"] == 9
    assert int(s.optimizer._adamw.state[s.model.b]["step"]) == 9


def test_first_step_after_fill_is_gradient_sized():
    # AdamW's first step from empty state moves each element by about lr (|m_hat / sqrt(v_hat)| = 1); after a fill
    # on noisy gradients of zero mean it moves by less
    def first_move(n_fill):
        s = _Stub(seed=0)
        it = _batches(seed=2)
        if n_fill:
            s._fill_optimizer_moments(it, n_fill)
        b0 = s.model.b.detach().clone()
        loss, _, _ = s._batch_loss(next(it))
        s.optimizer.zero_grad(); loss.backward(); s.optimizer.step()
        return (s.model.b.detach() - b0).abs().mean().item()
    fresh, filled = first_move(0), first_move(50)
    assert fresh > 0.9e-2 * 0.9, fresh            # about lr = 1e-2 per element
    assert filled < 0.6 * fresh, (fresh, filled)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
