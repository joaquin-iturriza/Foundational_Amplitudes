"""CPU check of MuonAdamW.adamw_lr_mult (fine_tune.adamw_warmup_steps, D303 step 9): at 0 the AdamW parameters stay
bit-identical while the Muon matrix moves as without it; at 1 (the default) a step equals the unmodified optimizer's;
the scheduler's lr is restored exactly after a scaled step; both multipliers at 0 freeze everything.
    python tests/test_adamw_warmup.py        (or pytest tests/test_adamw_warmup.py)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_muon_warmup import _setup  # noqa: E402


def test_mult_zero_freezes_adamw_only():
    W, b, opt = _setup()
    W0, b0 = W.detach().clone(), b.detach().clone()
    opt.adamw_lr_mult = 0.0
    opt.step()
    assert torch.equal(b.detach(), b0)
    W2, b2, opt2 = _setup()
    opt2.step()
    assert torch.equal(W.detach(), W2.detach()) and not torch.equal(W.detach(), W0)


def test_mult_one_is_default():
    W, b, opt = _setup(); opt.adamw_lr_mult = 1.0; opt.step()
    W2, b2, opt2 = _setup(); opt2.step()
    assert torch.equal(W.detach(), W2.detach()) and torch.equal(b.detach(), b2.detach())


def test_lr_restored():
    W, b, opt = _setup()
    before = [g["lr"] for g in opt.param_groups]
    opt.adamw_lr_mult = 0.3
    opt.step()
    assert [g["lr"] for g in opt.param_groups] == before
    assert [g["lr"] for g in opt._adamw.param_groups] == before[opt._n_muon:]


def test_both_zero_freeze_all():
    W, b, opt = _setup()
    W0, b0 = W.detach().clone(), b.detach().clone()
    opt.muon_lr_mult = 0.0; opt.adamw_lr_mult = 0.0
    opt.step()
    assert torch.equal(W.detach(), W0) and torch.equal(b.detach(), b0)


if __name__ == "__main__":
    for f in (test_mult_zero_freezes_adamw_only, test_mult_one_is_default, test_lr_restored, test_both_zero_freeze_all):
        f(); print("ok", f.__name__)
