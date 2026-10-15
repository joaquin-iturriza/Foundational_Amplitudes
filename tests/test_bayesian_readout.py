#!/usr/bin/env python
"""BNN last layer (model.net.bayesian_last_layer; models/bayesian_readout.py, the Heidelberg VBLinear as a μP readout).

Part 1 (torch + mup + lloca.mup; any CPU):
  - the layer: init (mu_w ~ N(0, 1/fan_in), log sigma^2 ~ N(-9, 0.001), bias 0), KL equal to
    torch.distributions' closed form, train-mode output mean -> posterior-mean output and variance
    sum_i x_i'^2 sigma_i^2 (local reparametrization), eval = posterior mean, sample_eval = one held weight draw;
  - μP: inside a @mup_parametrized toy backbone the weight and log sigma^2 carry readout infshapes (fan-in
    infinite, vector-like, so MuAdam leaves their lr alone), the weight is rescaled by sqrt(width_mult) and
    log sigma^2 is not, the forward is MuReadout's with the posterior-mean weight, and a sampled weight w acts as
    MuReadout with weight w.
Part 2 (needs the full stack: models.lloca, experiment; a site, scripts/job_test_bnn.sh):
  - bayesian_last_layer=False builds the model of the code without the flag (same seed, bit for bit);
  - the flag replaces net.linear_out only; eval output is deterministic and train output stochastic;
  - experiment._reg_term adds KL/N_train to lambda*reg only with a Bayesian readout; _reg_params is
    model.parameters() without it and skips logsig2_w with it.

Run:  python tests/test_bayesian_readout.py
"""
import math
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from torch import nn  # noqa: E402
from mup import MuReadout, MuAdam  # noqa: E402
from lloca.mup import mup_parametrized  # noqa: E402

from models.bayesian_readout import MuBayesianReadout, bayesian_readouts  # noqa: E402


def test_init_and_kl():
    torch.manual_seed(0)
    layer = MuBayesianReadout(512, 3)
    w, ls = layer.weight.detach(), layer.logsig2_w.detach()
    assert abs(w.std().item() * math.sqrt(512) - 1) < 0.05, w.std()
    assert abs(ls.mean().item() + 9) < 1e-3 and ls.std().item() < 2e-3
    assert torch.all(layer.bias == 0)
    for prec in (1.0, 4.0):
        layer.prior_prec = prec
        q = torch.distributions.Normal(w, ls.exp().sqrt())
        p = torch.distributions.Normal(torch.zeros_like(w), torch.full_like(w, prec ** -0.5))
        ref = torch.distributions.kl_divergence(q, p).sum()
        assert torch.allclose(layer.kl().detach(), ref, rtol=1e-5), (layer.kl(), ref)
    print("init N(0,1/fan_in), log sigma^2 N(-9,0.001), bias 0; KL = closed form: ok")


@mup_parametrized
class _Toy(nn.Module):
    DEFAULT_MUP_SHAPES = ({"width": 16}, {"width": 64})

    def __init__(self, width, bayes=True, *, parametrization="mup", mup_base_shapes=None, mup_delta_shapes=None):
        super().__init__()
        self.parametrization = parametrization
        self.linear_in = nn.Linear(5, width)
        self.linear_out = MuBayesianReadout(width, 2) if bayes else MuReadout(width, 2)

    def forward(self, x):
        return self.linear_out(torch.tanh(self.linear_in(x)))


def test_mup():
    torch.manual_seed(1)
    width = 256
    net = _Toy(width)
    ro = net.linear_out
    wm = ro.width_mult()
    assert wm == width / 16, wm
    for p in (ro.weight, ro.logsig2_w):
        assert p.infshape.ninf() == 1 and p.infshape[1].isinf() and not p.infshape[0].isinf(), p.infshape
    # _rescale_parameters: weight std sqrt(wm/fan_in) = 1/sqrt(base fan-in); log sigma^2 untouched
    assert abs(ro.weight.std().item() * math.sqrt(16) - 1) < 0.1, ro.weight.std()
    assert abs(ro.logsig2_w.mean().item() + 9) < 1e-3
    # MuAdam: the readout's parameters keep the base lr (vector-like), the hidden matrix does not exist here,
    # linear_in (finite fan-in, infinite fan-out) is vector-like too
    opt = MuAdam(net.parameters(), lr=1e-3)
    lrs = {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}
    assert lrs[id(ro.weight)] == 1e-3 and lrs[id(ro.logsig2_w)] == 1e-3, lrs
    # eval = MuReadout with the posterior-mean weight
    plain = _Toy(width, bayes=False)
    plain.load_state_dict({k: v for k, v in net.state_dict().items() if not k.endswith("logsig2_w")})
    x = torch.randn(64, 5)
    net.eval(); plain.eval()
    assert torch.allclose(net(x), plain(x), atol=1e-6)
    # a held weight draw w acts as MuReadout with weight w
    ro.sample_eval = True
    out = net(x)
    assert torch.equal(out, net(x)), "the draw is held until reset_random"
    w_draw = ro.weight + ro._logsig2().exp().sqrt() * ro.random
    with torch.no_grad():
        plain.linear_out.weight.copy_(w_draw)
    assert torch.allclose(out, plain(x), atol=1e-6)
    ro.reset_random()
    assert not torch.equal(out, net(x))
    ro.sample_eval = False
    print(f"μP: readout infshapes, weight rescaled by sqrt({wm:g}), log sigma^2 not, MuAdam lr unscaled, "
          "eval = MuReadout(mu_w), a draw = MuReadout(w): ok")


def test_local_reparametrization():
    """train-mode outputs: mean x' mu^T + b, variance x'^2 sigma^2 (x' = x / width_mult)."""
    torch.manual_seed(2)
    net = _Toy(64)
    ro = net.linear_out
    with torch.no_grad():
        ro.logsig2_w.fill_(-2.0)        # a visible posterior width
    h = torch.tanh(net.linear_in(torch.randn(3, 5))).detach()
    ro.train()
    S = 200000
    outs = ro(h.unsqueeze(0).expand(S, -1, -1))
    ro.eval()
    mean_ref = ro(h)
    xs = h / ro.width_mult()
    var_ref = xs.pow(2) @ ro.logsig2_w.exp().t()
    assert torch.allclose(outs.mean(0), mean_ref, atol=4 * (var_ref.max().sqrt() / math.sqrt(S)).item())
    assert torch.allclose(outs.var(0), var_ref + 1e-8, rtol=0.02)
    print("training forward: sample mean -> posterior-mean output, variance sum x'^2 sigma^2: ok")


# ------------------------------------------------------------------------------------------ part 2 (full stack)
def _full_stack():
    try:
        from models.lloca import LLOCAMuPTransformer  # noqa: F401  (lloca, torch_geometric)
        import experiment  # noqa: F401
        return True
    except Exception as exc:
        print(f"part 2: skipped ({type(exc).__name__}: {exc})")
        return False


def _model(seed=0, **kw):
    from models.lloca import LLOCAMuPTransformer
    torch.manual_seed(seed)
    return LLOCAMuPTransformer(num_scalars=4, hidden_channels_mlp=16, num_layers_mlp=2, in_channels=8,
                               attn_reps="8x0n+2x1n", out_channels=1, num_blocks=2, num_heads=4, **kw)


def _inputs(device):
    torch.manual_seed(7)
    lens = [4, 6, 5]
    ptr = torch.tensor([0, 4, 10, 15], device=device)
    p3 = torch.randn(15, 3)
    E = (p3.pow(2).sum(-1) + torch.rand(15)).sqrt()
    return torch.cat([E[:, None], p3], -1).to(device), torch.randn(15, 4, device=device), ptr, lens


def test_full_model():
    import experiment as E
    device = "cuda" if torch.cuda.is_available() else "cpu"
    old, off, on = _model().to(device), _model(bayesian_last_layer=False).to(device), \
        _model(bayesian_last_layer=True).to(device)
    so, sf = old.state_dict(), off.state_dict()
    assert so.keys() == sf.keys() and all(torch.equal(so[k], sf[k]) for k in so)
    assert type(off.net.linear_out) is MuReadout and type(on.net.linear_out) is MuBayesianReadout
    extra = set(on.state_dict()) - set(so)
    assert extra == {"net.linear_out.logsig2_w"}, extra
    fm, sc, ptr, _ = _inputs(device)
    on.eval()
    with torch.no_grad():
        a, b = on(fm, sc, 0.0, 1.0, ptr), on(fm, sc, 0.0, 1.0, ptr)
    assert torch.equal(a, b), "eval (posterior mean) is deterministic"
    on.train()
    with torch.no_grad():
        c = on(fm, sc, 0.0, 1.0, ptr)
    assert not torch.equal(a, c), "training forward samples"
    # _reg_params / _reg_term
    assert [id(p) for p in E._reg_params(off)] == [id(p) for p in off.parameters()]
    assert all(p is not on.net.linear_out.logsig2_w for p in E._reg_params(on))
    from omegaconf import OmegaConf
    stub = types.SimpleNamespace(_n_train_events=1000, cfg=OmegaConf.create(
        {"training": {"regularization": "L2", "regularization_lambda": 1e-3}}))
    E.AmplitudeExperiment._init_regularization(stub)
    rt = lambda m: E.AmplitudeExperiment._reg_term(stub, m)  # noqa: E731
    assert torch.equal(rt(off), 1e-3 * stub.regularization(off))
    stub._bnn_cache = None
    ref = 1e-3 * stub.regularization(on) + on.net.linear_out.kl() / 1000
    assert torch.allclose(rt(on), ref)
    print("full model: off = no flag (bit for bit), only linear_out replaced, eval deterministic, train sampled; "
          "_reg_term = lambda*L2 (+ KL/N_train with the flag), log sigma^2 outside L2: ok")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    test_init_and_kl()
    test_mup()
    test_local_reparametrization()
    if _full_stack():
        test_full_model()
    print("all passed")
