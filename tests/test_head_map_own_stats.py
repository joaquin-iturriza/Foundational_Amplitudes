"""fine_tune.target_stats=own: the parent's readout is rewritten at load so the fine-tune's step-0 prediction of y
(own stats: m_o + s_o * mu) equals the parent's zero-shot (m_p + s_p * mu_parent); sigma likewise where softplus ~ exp
(BaseExperiment._map_head_to_own_stats, pretrain_data_saving step 9)."""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import gzip
import json
import os
import sys
import tempfile

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_experiment import BaseExperiment  # noqa: E402

MP, SP, MO, SO = 0.22, 5.47, 0.72, 0.74   # parent / own target mean and std (e+e- > t t~ tree against r4)


class _Toy(torch.nn.Module):
    """A readout as the model's: muP-style input scaling, per-particle rows [mu, sigma], mean pooling over particles,
    sigma = softplus(pooled)."""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Module()
        self.net.net = torch.nn.Module()
        self.net.net.linear_out = torch.nn.Linear(6, 2)

    def forward(self, h):                     # h: (events, particles, 6)
        z = self.net.net.linear_out(0.5 * h / 4.0).mean(1)
        return z[:, 0], F.softplus(z[:, 1])


class _Stub:
    _load_pretrained_weights = BaseExperiment._load_pretrained_weights
    _map_head_to_own_stats = BaseExperiment._map_head_to_own_stats

    def __init__(self):
        self.cfg = OmegaConf.create({"seed": 42, "training": {"loss": "HETEROSC"}, "model": {"net": {}},
                                     "fine_tune": {"target_stats": "own", "grow_width": False, "shrink_perturb": None}})
        self.model = _Toy()
        self.device = "cpu"
        self.prepd_mean, self.prepd_std = [MO], [SO]


def test_step0_is_parent_zero_shot():
    torch.manual_seed(0)
    parent = _Toy()
    with torch.no_grad():
        parent.net.net.linear_out.bias[1] = -9.0          # a trained parent's small sigma
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "models"))
        json.dump({"prepd_mean": [MP] * 3, "prepd_std": [SP] * 3}, open(os.path.join(tmp, "data_stats.json"), "w"))
        OmegaConf.save(OmegaConf.create({"training": {"loss": "HETEROSC"}}), os.path.join(tmp, "config.yaml"))
        p = os.path.join(tmp, "models", "m.pt.gz")
        with gzip.open(p, "wb") as f:
            torch.save({"model": parent.state_dict()}, f)
        s = _Stub()
        s._load_pretrained_weights(p)
    h = torch.randn(64, 5, 6)
    with torch.no_grad():
        mu_p, sg_p = parent(h)
        mu_o, sg_o = s.model(h)
    torch.testing.assert_close(MO + SO * mu_o, MP + SP * mu_p, rtol=0, atol=1e-5)    # y exactly the parent's
    torch.testing.assert_close(SO * sg_o, SP * sg_p, rtol=1e-3, atol=0)              # sigma in y units


if __name__ == "__main__":
    test_step0_is_parent_zero_shot()
    print("ok")
