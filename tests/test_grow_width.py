"""CPU checks of the function-preserving width growth (models/grow_width.py; plan warm_start step 4, D285): a μP
LLoCa het model at num_heads 2 grown to 4 gives the parent's outputs (mu and sigma) on random events to float rounding
without noise (1e-5 x the largest output) and within 1e-3 x the largest output with the default
symmetry-breaking noise (the error is linear in the noise: 1.6e-3 on outputs up to 3.6 at noise 1e-3, 1.6e-4 at 1e-4); the noise makes the duplicated units differ;
a shape that is not a doubling is refused.
    .venv-cpu/bin/python -m pytest tests/test_grow_width.py
"""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from test_het_sign_head import build, batch, forward  # noqa: E402
from models.grow_width import grow_state  # noqa: E402


def trained_like(model, seed=3):
    """The parent's weights moved off their init (a zero-init readout or bias would hide a wrong mapping)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return model


def grown(noise):
    parent, _ = build(False, seed=7, heads=2)
    trained_like(parent)
    child, _ = build(False, seed=11, heads=4)
    sd, names = grow_state(parent.state_dict(), child, parent_heads=2, noise=noise, seed=0)
    child.load_state_dict(sd)
    return parent.eval(), child.eval(), names


@pytest.mark.parametrize("noise, tol", [(0.0, 1e-5), (1e-3, 1e-3)])
def test_function_preserved(noise, tol):
    parent, child, names = grown(noise)
    assert any("qkv_linear" in n for n in names) and any("mlp" in n for n in names)
    assert any("linear_out" in n for n in names)
    b = batch()
    with torch.no_grad():
        torch.manual_seed(0); yp = forward(parent, b)
        torch.manual_seed(0); yc = forward(child, b)
    err = (yp - yc).abs().max().item() / yp.abs().max().item()
    assert err < tol, err


def test_noise_breaks_symmetry():
    _, child, _ = grown(1e-3)
    w = dict(child.named_parameters())
    k = next(n for n in w if n.endswith("mlp.0.weight"))
    half = w[k].shape[0] // 2
    assert not torch.equal(w[k][:half], w[k][half:])


def test_refuses_non_doubling():
    parent, _ = build(False, seed=7, heads=2)
    child, _ = build(False, seed=11, heads=3)
    with pytest.raises(ValueError):
        grow_state(parent.state_dict(), child, parent_heads=2)


def test_loader_grows(tmp_path):
    """base_experiment._load_pretrained_weights with fine_tune.grow_width: the parent's num_heads read from its run's
    config.yaml, the grown model computes the parent's function (HETEROSC parent and child, readout rows unchanged)."""
    import types
    from omegaconf import OmegaConf
    from base_experiment import BaseExperiment
    parent, _ = build(False, seed=7, heads=2)
    trained_like(parent)
    run = tmp_path / "parent"
    (run / "models").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"model": {"net": {"num_heads": 2}}, "training": {"loss": "HETEROSC"}}),
                   run / "config.yaml")
    torch.save({"model": parent.state_dict()}, run / "models" / "model_run0_best.pt")
    child, _ = build(False, seed=11, heads=4)
    e = BaseExperiment.__new__(BaseExperiment)
    e.cfg = OmegaConf.create({"seed": 42, "training": {"loss": "HETEROSC"},
                              "model": {"net": {"num_heads": 4, "sign_channel": False}},
                              "fine_tune": {"grow_width": True, "grow_noise": 0.0}})
    e.device = torch.device("cpu")
    e.model = child
    e._load_pretrained_weights(str(run / "models" / "model_run0_best.pt"))
    b = batch()
    with torch.no_grad():
        torch.manual_seed(0); yp = forward(parent.eval(), b)
        torch.manual_seed(0); yc = forward(child.eval(), b)
    assert (yp - yc).abs().max().item() / yp.abs().max().item() < 1e-5
