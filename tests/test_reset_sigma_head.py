"""CPU check of fine_tune.reset_sigma_head (plan finetune_mechanism, sigma-start arm): a HETEROSC fine-tune loads the
parent's trunk and mu row but keeps its own fresh-init sigma row; off, the parent's sigma row loads; an MSE run is
refused; the sigma stage's config clears the switch (else the stage would reset the run's trained sigma).
    python tests/test_reset_sigma_head.py        (or pytest tests/test_reset_sigma_head.py)
"""
import _tmpdir  # noqa: F401
import os
import sys
import tempfile

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_experiment import BaseExperiment  # noqa: E402


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Linear(3, 4)
        self.linear_out = torch.nn.Linear(4, 2)   # rows [mu, sigma]


class _Stub:
    _load_pretrained_weights = BaseExperiment._load_pretrained_weights

    def __init__(self, reset, loss="HETEROSC"):
        self.cfg = OmegaConf.create({"seed": 42, "training": {"loss": loss}, "model": {"net": {"out_channels": 1}},
                                     "fine_tune": {"reset_sigma_head": reset, "grow_width": False,
                                                   "shrink_perturb": None}})
        self.model = _Net()
        self.device = "cpu"


def _parent(tmp):
    os.makedirs(os.path.join(tmp, "models"))
    OmegaConf.save(OmegaConf.create({"training": {"loss": "HETEROSC"}}), os.path.join(tmp, "config.yaml"))
    sd = {k: torch.full_like(v, 7.0) for k, v in _Net().state_dict().items()}
    path = os.path.join(tmp, "models", "p.pt")
    torch.save({"model": sd, "ema": None}, path)
    return path


def test_reset_sigma_head():
    with tempfile.TemporaryDirectory() as tmp:
        p = _parent(tmp)
        s = _Stub(True)
        fresh = {k: v.clone() for k, v in s.model.state_dict().items()}
        s._load_pretrained_weights(p)
        sd = s.model.state_dict()
        assert torch.all(sd["body.weight"] == 7.0) and torch.all(sd["body.bias"] == 7.0), "trunk not the parent's"
        assert torch.all(sd["linear_out.weight"][0] == 7.0) and sd["linear_out.bias"][0] == 7.0, "mu not the parent's"
        assert torch.equal(sd["linear_out.weight"][1], fresh["linear_out.weight"][1]), "sigma weight not fresh"
        assert torch.equal(sd["linear_out.bias"][1], fresh["linear_out.bias"][1]), "sigma bias not fresh"
        s = _Stub(False); s._load_pretrained_weights(p)
        assert all(torch.all(v == 7.0) for v in s.model.state_dict().values()), "off: parent not loaded whole"
        try:
            _Stub(True, loss="MSE")._load_pretrained_weights(p)
        except RuntimeError as e:
            assert "HETEROSC" in str(e)
        else:
            raise AssertionError("reset_sigma_head on an MSE run was not refused")


def test_sigma_stage_clears_reset_sigma_head():
    from base_experiment import clear_start_only
    ft = OmegaConf.create({"reset_sigma_head": True})
    clear_start_only(ft)
    assert ft.reset_sigma_head is False


if __name__ == "__main__":
    test_reset_sigma_head()
    test_sigma_stage_clears_reset_sigma_head()
    print("OK")
