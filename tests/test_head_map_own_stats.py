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


def test_no_map_when_parent_predicts_another_quantity():
    """A tree parent (target_over_born false) under a V/B run: readout loaded unchanged."""
    torch.manual_seed(0)
    parent = _Toy()
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "models"))
        json.dump({"prepd_mean": [MP], "prepd_std": [SP]}, open(os.path.join(tmp, "data_stats.json"), "w"))
        OmegaConf.save(OmegaConf.create({"training": {"loss": "HETEROSC"}}), os.path.join(tmp, "config.yaml"))
        p = os.path.join(tmp, "models", "m.pt.gz")
        with gzip.open(p, "wb") as f:
            torch.save({"model": parent.state_dict()}, f)
        s = _Stub()
        s.cfg.data = {"target_over_born": True}
        s._load_pretrained_weights(p)
    for k, v in parent.state_dict().items():
        torch.testing.assert_close(s.model.state_dict()[k], v, rtol=0, atol=0)


def _asym_model(sd):
    """A small model.input_asym_factor LLoCa (HETEROSC, rows [mu, alpha, sigma]) with its target std `sd`, as
    AmplitudeExperiment._post_instantiate_model leaves it."""
    from models.lloca import LLOCAMuPTransformer
    from particle_ids import GLOBAL_PROPERTY_MATRIX as G
    from wrappers import AmplitudeLLoCaWrapper
    net = LLOCAMuPTransformer(num_scalars=16, hidden_channels_mlp=16, num_layers_mlp=2, in_channels=20,
                              attn_reps="8x0n+2x1n", out_channels=1, num_blocks=2, num_heads=2, loss="HETEROSC",
                              extra_main=1)
    w = AmplitudeLLoCaWrapper(net, token_size=0, d_particle_hidden=16, input_asym_factor=True)
    w.setup_particle_features(use_pids=False, property_matrix=G)
    w.asym_sd.fill_(sd)
    w.asym_out.copy_(torch.tensor([0.0, sd], dtype=torch.float64))
    return w.double().eval()


def test_asym_factor_step0_is_parent_zero_shot():
    """model.input_asym_factor under target_stats=own (kit_benchmark step 34): the map keeps the parent's readouts and
    unit and maps the output, so y = m + s mu is the parent's exactly, in the bulk and in a soft and a collinear limit
    (mapping row 0 scaled the limits' slopes by s_o/s_p); a parent checkpoint from before asym_out (no key) maps the
    same; a run rebuilt on its own stats loads unchanged."""
    import numpy as np
    from particle_ids import global_encode
    from test_input_reps import _uug
    tok = torch.tensor(global_encode(np.array([-11, 11, 2, -2, 21])))
    evs = [_uug(0.9, 0.8), _uug(1 - 0.4e-6, 1 - 0.6e-6), _uug(0.7 + 1e-7 * 0.3, 1 - 1e-7)]
    p, ptr = torch.tensor(np.concatenate(evs)), torch.tensor([0, 5, 10, 15])
    y = lambda m, w, mean: (mean + w.asym_out[1] * m(p, torch.cat([tok] * 3), 0.0, 1.0, ptr)[:, 0]).detach()
    for old in (False, True):
        torch.manual_seed(1)
        parent = _asym_model(SP)
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "models"))
            json.dump({"prepd_mean": [MP], "prepd_std": [SP]}, open(os.path.join(tmp, "data_stats.json"), "w"))
            OmegaConf.save(OmegaConf.create({"training": {"loss": "HETEROSC"}}), os.path.join(tmp, "config.yaml"))
            sd = {k: v for k, v in parent.state_dict().items() if not (old and k == "asym_out")}
            pt = os.path.join(tmp, "models", "m.pt.gz")
            with gzip.open(pt, "wb") as f:
                torch.save({"model": sd}, f)
            s = _Stub()
            torch.manual_seed(2)
            s.model = _asym_model(SO)
            s._load_pretrained_weights(pt)
        yp, yo = y(parent, parent, MP), y(s.model, s.model, MO)
        torch.testing.assert_close(yo, yp, rtol=0, atol=1e-9)
        torch.testing.assert_close(s.model.asym_sd, parent.asym_sd)
    with tempfile.TemporaryDirectory() as tmp:    # the fine-tune rebuilt on its own checkpoint and stats
        os.makedirs(os.path.join(tmp, "models"))
        json.dump({"prepd_mean": [MO], "prepd_std": [SO]}, open(os.path.join(tmp, "data_stats.json"), "w"))
        OmegaConf.save(OmegaConf.create({"training": {"loss": "HETEROSC"}}), os.path.join(tmp, "config.yaml"))
        pt = os.path.join(tmp, "models", "m.pt.gz")
        with gzip.open(pt, "wb") as f:
            torch.save({"model": s.model.state_dict()}, f)
        r = _Stub()
        r.model = _asym_model(SO)
        r._load_pretrained_weights(pt)
    torch.testing.assert_close(y(r.model, r.model, MO), yo, rtol=0, atol=0)




if __name__ == "__main__":
    test_step0_is_parent_zero_shot()
    test_no_map_when_parent_predicts_another_quantity()
    test_asym_factor_step0_is_parent_zero_shot()
    print("ok")


def test_affine_parent_maps_at_the_targets_multiplicity():
    """A data.affine_multiplicity_std parent (one mean per particle count, mu_N = a + b N): the map uses mu_N at this
    run's process's N (kit_benchmark step 20, E9497)."""
    torch.manual_seed(0)
    parent = _Toy()
    a, b, n = -34.3, 7.95, 4                  # parent means -18.4 / -10.5 / -2.5 at N = 2 / 3 / 4 (the fair-NLO parent)
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "models"))
        json.dump({"prepd_mean": [a + b * k for k in (2, 3, 4)], "prepd_std": [SP] * 3, "shared_standardization": True,
                   "affine_multiplicity": {"a": a, "b": b, "sigma": SP}}, open(os.path.join(tmp, "data_stats.json"), "w"))
        OmegaConf.save(OmegaConf.create({"training": {"loss": "HETEROSC"}}), os.path.join(tmp, "config.yaml"))
        p = os.path.join(tmp, "models", "m.pt.gz")
        with gzip.open(p, "wb") as f:
            torch.save({"model": parent.state_dict()}, f)
        s = _Stub()
        s._amp_npart = {"kit_ee_tt__virt_finite": n}
        s._load_pretrained_weights(p)
    h = torch.randn(64, 5, 6)
    with torch.no_grad():
        mu_p, _ = parent(h)
        mu_o, _ = s.model(h)
    torch.testing.assert_close(MO + SO * mu_o, (a + b * n) + SP * mu_p, rtol=0, atol=1e-5)
