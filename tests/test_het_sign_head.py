"""CPU checks of the sign head under HETEROSC (D287; models/lloca.py sign_channel, experiment.init_data):
the readout has rows [mu, sign, sigma] and the wrapper returns [mu, sign logit, sigma > 0]; at one seed the backbone,
the mu and sigma rows, the wrapper's particle encoder and the global RNG after construction are bit-equal to the het
model without the head; the sign BCE's gradient reaches the sign row (and only through sign_loss_weight); with
sign_loss_weight 0 the loss equals the het magnitude model's on a fixed batch (to float rounding); the sigma stage
(_freeze_all_but_sigma_head) leaves the mu and sign rows and the backbone bit-equal while the sigma row trains.
    .venv-cpu/bin/python -m pytest tests/test_het_sign_head.py
"""
import os
import sys
import types

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiment import AmplitudeExperiment as E  # noqa: E402
from models.lloca import LLOCAMuPTransformer  # noqa: E402
from wrappers import AmplitudeLLoCaWrapper  # noqa: E402
from particle_ids import GLOBAL_PROPERTY_MATRIX  # noqa: E402
from lloca.mup import finalize as mup_finalize  # noqa: E402

D_PARTICLE_HIDDEN = 16
N_ORDER = 2
NUM_SCALARS = D_PARTICLE_HIDDEN + N_ORDER


def build(sign, seed=7, heads=2):
    torch.manual_seed(seed)
    net = LLOCAMuPTransformer(num_scalars=NUM_SCALARS, hidden_channels_mlp=128, num_layers_mlp=2,
                              in_channels=NUM_SCALARS + 4, attn_reps="8x0n+2x1n", out_channels=1, num_blocks=2,
                              num_heads=heads, loss="HETEROSC", sigma_after_pool=True, sign_channel=sign)
    model = AmplitudeLLoCaWrapper(net, token_size=0, d_particle_hidden=D_PARTICLE_HIDDEN)
    model.setup_particle_features(use_pids=False, property_matrix=GLOBAL_PROPERTY_MATRIX)
    mup_finalize(model)
    return model, torch.random.get_rng_state()


def batch(n_events=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(3, 6, (n_events,), generator=g)
    N = int(counts.sum())
    ptr = torch.zeros(n_events + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(counts, 0)
    p = 3.0 * torch.randn(N, 3, generator=g)
    m = 0.5 + torch.rand(N, 1, generator=g)
    fm = torch.cat([(p.pow(2).sum(-1, keepdim=True) + m.pow(2)).sqrt(), p], -1)
    pt = torch.randint(0, GLOBAL_PROPERTY_MATRIX.shape[0], (N,), generator=g)
    y = torch.randn(n_events, 1, generator=g)
    sgn = (torch.rand(n_events, 1, generator=g) > 0.5).float()
    pids = torch.arange(n_events) % 4
    return dict(fm=fm, pt=pt, ptr=ptr, order=torch.zeros(n_events, N_ORDER), y=y, sgn=sgn, pids=pids)


def forward(model, b):
    return model(b["fm"], b["pt"], 0.0, 1.0, b["ptr"], order_labels=b["order"])


def stub(model, sign, w=1.0, sigma_only=False):
    s = types.SimpleNamespace()
    s.cfg = OmegaConf.create({"training": {"loss": "HETEROSC", "loss_aggregation": "mean", "sign_loss_weight": w,
                                           "sign_near_weight": 1.0, "heterosc_sigma_only": sigma_only},
                              "model": {"net": {"out_channels": 1, "sign_channel": sign}},
                              "data": {"dataset": ["p0", "p1", "p2", "p3"]}})
    s.n_datasets = 4
    s._sign_head = sign
    s._proc_signed = np.array([0, 1, 0, 1], dtype=np.float32)
    s.model = model
    s._per_event_loss = types.MethodType(E._per_event_loss, s)
    return s


def loss(s, out, b):
    """The experiment's split (_forward_lloca: sigma = last out_shape channels, then the sign = the last one) and
    loss (_aggregate_per_process_loss)."""
    sigma, y_pred = out[..., -1:], out[..., :-1]
    sign = None
    y = b["y"]
    if s._sign_head:
        sign = (y_pred[..., -1:], b["sgn"])
        y_pred = y_pred[..., :-1]
    return E._aggregate_per_process_loss(s, y_pred, y, b["pids"], "mean", sigma=sigma, sign=sign)


def test_readout_rows_and_split():
    model, _ = build(True)
    assert model.net.net.linear_out.weight.shape[0] == 3
    out = forward(model, batch())
    assert out.shape == (24, 3)
    assert (out[:, 2] > 0).all()                       # sigma (softplus) is the last column


def test_init_equal_to_het_model():
    a, rng_a = build(False)
    b, rng_b = build(True)
    assert torch.equal(rng_a, rng_b), "the RNG after construction differs"
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    for k in sa:
        if "linear_out" in k:
            assert torch.equal(sb[k][0], sa[k][0]) and torch.equal(sb[k][2], sa[k][1]), k
        else:
            assert torch.equal(sa[k], sb[k]), k
    # the μP readout scaling is the same on the grown readout
    assert a.net.net.linear_out.weight.infshape.width_mult() == b.net.net.linear_out.weight.infshape.width_mult()


def test_weight_zero_equals_het_loss_and_bce_reaches_sign_row():
    bt = batch()
    a, _ = build(False)
    b, _ = build(True)
    la = loss(stub(a, False), forward(a, bt), bt)
    lb0 = loss(stub(b, True, w=0.0), forward(b, bt), bt)
    # equal to float rounding: the backbone and the mu and sigma rows are bit-equal (test_init_equal_to_het_model),
    # but a 3-row readout's matmul may sum in another order than a 2-row one
    assert torch.allclose(la, lb0, rtol=1e-6, atol=0), (float(la), float(lb0))
    oa, ob = forward(a, bt), forward(b, bt)
    assert torch.allclose(oa[:, 0], ob[:, 0], rtol=1e-6, atol=1e-7)
    assert torch.allclose(oa[:, 1], ob[:, 2], rtol=1e-6, atol=1e-7)
    # the BCE gradient reaches the sign row with weight 1, not with weight 0
    for w, expect in ((1.0, True), (0.0, False)):
        b.zero_grad(set_to_none=True)
        loss(stub(b, True, w=w), forward(b, bt), bt).backward()
        g = b.net.net.linear_out.weight.grad[1]
        assert bool(g.abs().sum() > 0) == expect, (w, float(g.abs().sum()))
    lb1 = loss(stub(b, True, w=1.0), forward(b, bt), bt)
    assert float(lb1) > float(lb0)


def test_sigma_stage_trains_sigma_row_only():
    bt = batch(seed=3)
    b, _ = build(True)
    s = stub(b, True, sigma_only=True)
    before = {k: v.detach().clone() for k, v in b.named_parameters()}   # parameters (a buffer flag flips at 1st use)
    E._freeze_all_but_sigma_head(types.SimpleNamespace(cfg=s.cfg, model=b))
    opt = torch.optim.Adam([p for p in b.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        loss(s, forward(b, bt), bt).backward()
        opt.step()
    after = dict(b.named_parameters())
    for k in before:
        if "linear_out" in k:
            assert torch.equal(after[k][:2], before[k][:2]), k          # mu and sign rows
            assert not torch.equal(after[k][2], before[k][2]), k        # sigma row trained
        else:
            assert torch.equal(after[k], before[k]), k


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
