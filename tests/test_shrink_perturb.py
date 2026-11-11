"""CPU check of BaseExperiment._shrink_perturb (fine_tune.shrink_perturb, plan warm_start step 14): every parameter
becomes lam * loaded + gamma * the same parameter of a fresh model built at seed + 1; the fresh draw differs from the
run's own init at its seed, the global RNG stream is left as it was, and a parameter the fresh model lacks raises.
    python tests/test_shrink_perturb.py        (or pytest tests/test_shrink_perturb.py)
"""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import base_experiment  # noqa: E402
from base_experiment import BaseExperiment  # noqa: E402


class _Net(torch.nn.Module):
    def __init__(self, extra=False):
        super().__init__()
        self.lin = torch.nn.Linear(6, 4)
        self.norm = torch.nn.LayerNorm(4)
        if extra:
            self.more = torch.nn.Linear(4, 1)


class _Stub:
    _shrink_perturb = BaseExperiment._shrink_perturb

    def __init__(self, seed, extra_fresh=False):
        self.cfg = OmegaConf.create({"seed": seed, "model": {"extra": extra_fresh}, "training": {"ema_decay": 0.9}})
        torch.manual_seed(seed)
        self.model = _Net()
        self.ema = None

    def _post_instantiate_model(self, m):
        pass

    def _is_mup_model(self):
        return False


def _instantiate(cfg):
    return _Net(extra=False) if not cfg.get("extra") else _NetMissing()


class _NetMissing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(6, 4)   # no norm: the run's norm parameters have no fresh counterpart


def test_shrink_perturb():
    base_experiment.instantiate = _instantiate
    s = _Stub(seed=42)
    own_init = {n: p.detach().clone() for n, p in s.model.named_parameters()}
    with torch.no_grad():   # the loaded parent: anything other than the init
        for p in s.model.parameters():
            p.add_(1.0)
    loaded = {n: p.detach().clone() for n, p in s.model.named_parameters()}
    torch.manual_seed(43)
    fresh = dict(_Net().named_parameters())
    torch.manual_seed(7)
    before = torch.random.get_rng_state()
    before_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    s._shrink_perturb(0.4, 0.1)
    assert torch.equal(torch.random.get_rng_state(), before), "the global RNG stream moved"
    if before_cuda is not None:
        assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), before_cuda)), "a CUDA stream moved"
    for n, p in s.model.named_parameters():
        want = 0.4 * loaded[n] + 0.1 * fresh[n].detach()
        assert torch.allclose(p, want, atol=1e-6), n
    assert not torch.allclose(fresh["lin.weight"], own_init["lin.weight"]), "fresh draw equals the run's own init"
    s2 = _Stub(seed=42, extra_fresh=True)
    try:
        s2._shrink_perturb(0.4, 0.1)
    except RuntimeError as e:
        assert "norm" in str(e)
    else:
        raise AssertionError("a parameter without a fresh counterpart did not raise")


if __name__ == "__main__":
    test_shrink_perturb()
    print("ok")
