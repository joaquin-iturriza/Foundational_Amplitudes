"""CPU checks of optim/muon.py (MuonAdamW): the Newton-Schulz orthogonalisation, the update RMS (0.2 at muP's base
width), the muP scaling (1/sqrt of the width multiplier), decoupled weight decay, the AdamW side equal to torch's
AdamW, the scheduler reaching both sides, a state_dict round trip, and, where mup and lloca import (the CC venv), the
parameter split on the real LLoCa muP model: Muon on exactly the blocks' 32 hidden matrices.
    python tests/test_muon.py        (or pytest tests/test_muon.py)
"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optim.muon import MuonAdamW, newton_schulz  # noqa: E402


class _Inf:
    """A stand-in for mup's InfShape: ninf() and width_mult()."""
    def __init__(self, ninf, m):
        self._n, self._m = ninf, m

    def ninf(self):
        return self._n

    def width_mult(self):
        return self._m


def _param(shape, ninf, m, seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.nn.Parameter(torch.randn(*shape, generator=g))
    p.infshape = _Inf(ninf, m)
    return p


def _rms(x):
    return x.pow(2).mean().sqrt().item()


def test_orthogonalisation():
    for shape in [(64, 256), (256, 64), (128, 128)]:
        G = torch.randn(*shape, generator=torch.Generator().manual_seed(1))
        O = newton_schulz(G, 5)
        assert O.dtype == torch.float32 and O.shape == G.shape
        s = torch.linalg.svdvals(O)
        # NS5's quintic brings singular values to ~[0.7, 1.2], not exactly 1; a square Gaussian matrix's smallest few
        # start near 0 and stay below, so the check is on the bulk (90%) and the ceiling
        frac = ((s > 0.6) & (s < 1.25)).float().mean().item()
        assert frac > 0.9 and s.max() < 1.25, (shape, frac, s.min().item(), s.max().item())


def _one_step_update(m, shape=(128, 512), wd=0.0):
    p = _param(shape, 2, m, 0)
    p0 = p.detach().clone()
    p.grad = torch.randn(*shape, generator=torch.Generator().manual_seed(2))
    opt = MuonAdamW([p], lr=1.0, muon_weight_decay=wd)
    opt.step()
    return p.detach() - p0


def test_update_rms_and_mup_scaling():
    d1 = _one_step_update(1.0)
    assert 0.14 < _rms(d1) < 0.26, _rms(d1)          # 0.2 up to Newton-Schulz's spread of singular values
    d4 = _one_step_update(4.0)
    assert abs(_rms(d4) / _rms(d1) - 0.5) < 1e-5, _rms(d4) / _rms(d1)   # lr / sqrt(width multiplier)


def test_decoupled_weight_decay():
    p = _param((32, 64), 2, 1.0, 0)
    p0 = p.detach().clone()
    p.grad = torch.zeros_like(p)
    MuonAdamW([p], lr=0.5, muon_weight_decay=0.1).step()
    assert torch.allclose(p.detach(), p0 * (1 - 0.5 * 0.1), atol=1e-7)


def test_adamw_side_matches_torch_adamw():
    torch.manual_seed(0)
    vec = [_param((16,), 1, 4.0, 3), _param((8, 16), 0, 1.0, 4)]   # vector-like and finite: the baseline's lr
    ref = [torch.nn.Parameter(p.detach().clone()) for p in vec]
    opt = MuonAdamW([{"params": vec, "lr": 3e-3}], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    ropt = torch.optim.AdamW([{"params": ref, "lr": 3e-3}], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    for k in range(5):
        for a, b in zip(vec, ref):
            g = torch.randn(a.shape, generator=torch.Generator().manual_seed(10 + k))
            a.grad, b.grad = g.clone(), g.clone()
        opt.step()
        ropt.step()
    for a, b in zip(vec, ref):
        assert torch.equal(a.detach(), b.detach())


def test_scheduler_and_state_dict():
    mat, vec = _param((32, 64), 2, 4.0, 0), _param((32,), 1, 4.0, 1)
    opt = MuonAdamW([mat, vec], lr=1e-2)
    assert [g["use_muon"] for g in opt.param_groups] == [True, False]
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    for _ in range(3):
        mat.grad, vec.grad = torch.randn_like(mat), torch.randn_like(vec)
        opt.step()
        sch.step()
    lr = opt.param_groups[1]["lr"]
    assert lr < 1e-2 and opt._adamw.param_groups[0]["lr"] == lr and opt.param_groups[0]["lr"] == lr
    sd = copy.deepcopy(opt.state_dict())
    opt2 = MuonAdamW([mat, vec], lr=1e-2)
    opt2.load_state_dict(sd)
    assert opt2._adamw.param_groups[0] is opt2.param_groups[1]
    assert torch.equal(opt2.state[mat]["momentum_buffer"], opt.state[mat]["momentum_buffer"])
    assert torch.equal(opt2._adamw.state[vec]["exp_avg"], opt._adamw.state[vec]["exp_avg"])
    assert opt2.param_groups[1]["lr"] == lr


def test_real_lloca_split():
    try:
        from lloca.mup import finalize
        from tests.test_mup_coord_check import build_model
    except Exception as e:  # mup / lloca not installed here (the laptop): checked on CC
        print(f"  skipped: {e!r}")
        return
    model = build_model(8)
    finalize(model)
    opt = MuonAdamW([{"params": list(model.parameters()), "lr": 1e-2}], lr=1e-2)
    names = {id(p): n for n, p in model.named_parameters()}
    mu = [names[id(p)] for g in opt.param_groups if g["use_muon"] for p in g["params"]]
    rest = [p for g in opt.param_groups if not g["use_muon"] for p in g["params"]]
    assert len(mu) == 32 and all(".blocks." in n for n in mu), mu
    assert len(mu) + len(rest) == len(list(model.parameters()))
    assert all(g["lr"] == 1e-2 for g in opt.param_groups)
    ms = {p.infshape.width_mult() for g in opt.param_groups if g["use_muon"] for p in g["params"]}
    print(f"  Muon: {len(mu)} matrices, width multipliers {sorted(ms)}; e.g. {mu[:4]}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
