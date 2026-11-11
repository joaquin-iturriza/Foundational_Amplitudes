"""CPU checks of model.net.sign_branch (plan catalog_coverage step 73; models/transformer_lloca_mup.py sign_block,
base_experiment._clip_grad_norm): the het sign logit computed by a copy of the last transformer block fed the last
block's input and the frames detached.
(a) flag off: parameters, RNG after construction, outputs and gradients bit-equal to the code at REF (default trunk);
(b) flag on: every parameter but sign_block, and the RNG after construction, bit-equal to the flag-off het sign-head
    model at the same seed; sign_block equal to blocks[-1] at init, with the same μP infshapes;
(c) the sign logit's gradient reaches only sign_block and the readout's sign row (framesnet, encoders and trunk get
    exactly zero); mu's and sigma's never reach sign_block; with checkpoint_blocks and with LLOCA_FRAMES=repeat too;
(d) separate clipping: a huge sign gradient leaves every other gradient after clipping bit-equal to that without the
    sign term, and the sign group's norm at the max norm;
the sigma stage freezes sign_block and MuonAdamW steps with it frozen.
    .venv-cpu/bin/python -m pytest tests/test_sign_branch.py
"""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import importlib.util
import os
import subprocess
import sys
import tempfile
import types

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from base_experiment import BaseExperiment  # noqa: E402
from experiment import AmplitudeExperiment as E  # noqa: E402
from models.lloca import LLOCAMuPTransformer  # noqa: E402
from wrappers import AmplitudeLLoCaWrapper  # noqa: E402
from particle_ids import GLOBAL_PROPERTY_MATRIX  # noqa: E402
from lloca.mup import finalize as mup_finalize  # noqa: E402
from optim.muon import MuonAdamW  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_het_sign_head import NUM_SCALARS, D_PARTICLE_HIDDEN, batch, forward  # noqa: E402

REF = os.environ.get("SIGN_BRANCH_REF", "trunk")


def build(sign=True, branch=False, seed=7, heads=2, ckpt=False, net_cls=LLOCAMuPTransformer, blocks=2, k=None):
    torch.manual_seed(seed)
    kw = {"sign_branch": True} if branch else {}
    if k is not None:
        kw["sign_branch_blocks"] = k
    net = net_cls(num_scalars=NUM_SCALARS, hidden_channels_mlp=128, num_layers_mlp=2, in_channels=NUM_SCALARS + 4,
                  attn_reps="8x0n+2x1n", out_channels=1, num_blocks=blocks, num_heads=heads, loss="HETEROSC",
                  sigma_after_pool=True, sign_channel=sign, checkpoint_blocks=ckpt, **kw)
    model = AmplitudeLLoCaWrapper(net, token_size=0, d_particle_hidden=D_PARTICLE_HIDDEN)
    model.setup_particle_features(use_pids=False, property_matrix=GLOBAL_PROPERTY_MATRIX)
    mup_finalize(model)
    return model, torch.random.get_rng_state()


def _ref_net_cls():
    """LLOCAMuPTransformer of the models/ package at REF, loaded as its own package."""
    d = tempfile.mkdtemp()
    try:
        arc = subprocess.run(["git", "-C", ROOT, "archive", REF, "models"], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", d], input=arc, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        pytest.skip(f"no git ref {REF}: {e}")
    name = "models_ref_sign_branch"
    spec = importlib.util.spec_from_file_location(name, os.path.join(d, "models", "__init__.py"),
                                                  submodule_search_locations=[os.path.join(d, "models")])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[name] = pkg
    spec.loader.exec_module(pkg)
    return importlib.import_module(f"{name}.lloca").LLOCAMuPTransformer


@pytest.mark.parametrize("sign", [False, True])
def test_a_flag_off_equals_ref(sign):
    ref, rng_r = build(sign=sign, net_cls=_ref_net_cls())
    new, rng_n = build(sign=sign)
    assert torch.equal(rng_r, rng_n)
    sr, sn = ref.state_dict(), new.state_dict()
    assert list(sr) == list(sn)
    for k in sr:
        assert torch.equal(sr[k], sn[k]), k
    bt = batch()
    torch.manual_seed(1)
    o_r = forward(ref, bt)
    torch.manual_seed(1)
    o_n = forward(new, bt)
    assert torch.equal(o_r, o_n)
    o_r.square().sum().backward()
    o_n.square().sum().backward()
    for (k, p), (_, q) in zip(ref.named_parameters(), new.named_parameters()):
        assert (p.grad is None) == (q.grad is None) and (p.grad is None or torch.equal(p.grad, q.grad)), k
    assert not hasattr(new.net.net, "sign_block")


def test_b_init():
    a, rng_a = build(branch=False)
    b, rng_b = build(branch=True)
    assert torch.equal(rng_a, rng_b), "the RNG after construction differs"
    sa, sb = a.state_dict(), b.state_dict()
    extra = {k for k in sb if ".sign_block." in k}
    assert extra and set(sb) - extra == set(sa)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    net = b.net.net
    last = dict(net.blocks[-1].named_parameters())
    for n, p in net.sign_block.named_parameters():
        assert torch.equal(p, last[n]), n
        assert p.data_ptr() != last[n].data_ptr(), n               # a copy, not shared
        assert p.infshape.width_mult() == last[n].infshape.width_mult(), n
        assert p.infshape.ninf() == last[n].infshape.ninf(), n
    assert net.sign_block.attention.attention is net.attention    # the shared LLoCaAttention, as the trunk's


@pytest.mark.parametrize("blocks,k", [(3, 2), (3, 3), (2, 1)])
def test_b_init_k(blocks, k):
    """sign_branch_blocks k (step 80): RNG and every trunk parameter as the flag-off model; the branch's blocks equal
    the trunk's last k at init, copies with the same muP infshapes; at init the branch-on outputs equal the flag-off
    model's (the sign logit then reads the same features as the trunk's), so the branch takes block (blocks - k)'s
    input."""
    a, rng_a = build(branch=False, blocks=blocks)
    b, rng_b = build(branch=True, blocks=blocks, k=k)
    assert torch.equal(rng_a, rng_b)
    sa, sb = a.state_dict(), b.state_dict()
    extra = {n for n in sb if ".sign_block." in n}
    assert extra and set(sb) - extra == set(sa)
    for n in sa:
        assert torch.equal(sa[n], sb[n]), n
    net = b.net.net
    copies = [net.sign_block] if k == 1 else list(net.sign_block.blocks)
    assert len(copies) == k
    for c, t in zip(copies, net.blocks[blocks - k:]):
        tp = dict(t.named_parameters())
        for n, p in c.named_parameters():
            assert torch.equal(p, tp[n]) and p.data_ptr() != tp[n].data_ptr(), n
            assert p.infshape.width_mult() == tp[n].infshape.width_mult(), n
        assert c.attention.attention is net.attention
    bt = batch(seed=5)
    a.eval(); b.eval()
    with torch.no_grad():
        oa, ob = forward(a, bt), forward(b, bt)
    assert torch.allclose(oa, ob, rtol=1e-5, atol=1e-6), (oa - ob).abs().max()


@pytest.mark.parametrize("ckpt", [False, True])
def test_c_gradient_isolation_k(ckpt):
    """k = 2 of 3 blocks: the sign logit's gradient reaches only the branch and the readout's sign row; mu's and
    sigma's never reach the branch."""
    bt = batch(seed=5)
    m, _ = build(branch=True, ckpt=ckpt, blocks=3, k=2)
    out = forward(m, bt)
    out[:, 1].square().sum().backward()
    g = _grads(m)
    for n, t in g.items():
        if ".sign_block." in n:
            if n.endswith("weight"):
                assert t.abs().sum() > 0, n
            continue
        if n in SIGN_ROW:
            assert torch.count_nonzero(t[:1]) == 0 and torch.count_nonzero(t[2:]) == 0, n
            continue
        assert torch.count_nonzero(t) == 0, n
    m.zero_grad(set_to_none=True)
    out = forward(m, bt)
    (out[:, 0].square().sum() + out[:, 2].sum()).backward()
    g = _grads(m)
    for n, t in g.items():
        if ".sign_block." in n:
            assert torch.count_nonzero(t) == 0, n
    # the clip's sign group (base_experiment._clip_grad_norm: sign_block.parameters()) holds every block of the branch
    assert {id(p) for p in m.net.net.sign_block.parameters()} >= {id(p) for p in m.net.net.sign_block.blocks[1].parameters()}


def _grads(model):
    return {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for n, p in model.named_parameters()}


SIGN_ROW = ("net.net.linear_out.weight", "net.net.linear_out.bias")


@pytest.mark.parametrize("ckpt,frames", [(False, "broadcast"), (True, "broadcast"), (False, "repeat")])
def test_c_gradient_isolation(ckpt, frames, monkeypatch):
    monkeypatch.setenv("LLOCA_FRAMES", frames)
    bt = batch(seed=5)
    m, _ = build(branch=True, ckpt=ckpt)
    out = forward(m, bt)
    out[:, 1].square().sum().backward()                           # the sign logit alone
    g = _grads(m)
    fn_seen = False
    for n, t in g.items():
        if ".sign_block." in n:
            continue
        if n in SIGN_ROW:
            assert t[1].abs().sum() > 0, n
            assert torch.count_nonzero(t[:1]) == 0 and torch.count_nonzero(t[2:]) == 0, n
            continue
        fn_seen |= "framesnet" in n
        assert torch.count_nonzero(t) == 0, n
    assert fn_seen
    assert sum(float(t.abs().sum()) for n, t in g.items() if ".sign_block." in n) > 0
    # every sign_block matrix receives gradient
    for n, t in g.items():
        if ".sign_block." in n and n.endswith("weight"):
            assert t.abs().sum() > 0, n

    m.zero_grad(set_to_none=True)
    out = forward(m, bt)
    (out[:, 0].square().sum() + out[:, 2].sum()).backward()       # mu and sigma alone
    g = _grads(m)
    for n, t in g.items():
        if ".sign_block." in n:
            assert torch.count_nonzero(t) == 0, n
    for n in SIGN_ROW:
        assert torch.count_nonzero(g[n][1]) == 0, n
    assert sum(float(t.abs().sum()) for n, t in g.items() if "framesnet" in n) > 0
    # the live frames are back on the attention after the forward (the trunk's checkpoint recompute reads them)
    assert m.net.net.attention.frames.matrices.requires_grad


def _clip(model, max_norm=5.0):
    s = types.SimpleNamespace(model=model)
    return BaseExperiment._clip_grad_norm(s, max_norm)


def test_d_separate_clipping():
    bt = batch(seed=9)
    m, _ = build(branch=True)
    sd = {k: v.clone() for k, v in m.state_dict().items()}

    def run(sign_scale):
        m.load_state_dict(sd)
        m.zero_grad(set_to_none=True)
        torch.manual_seed(0)                                       # the forward draws random numbers
        out = forward(m, bt)
        loss = 1e3 * (out[:, 0].square().sum() + out[:, 2].sum()) + sign_scale * out[:, 1].square().sum()
        loss.backward()
        pre = _grads(m)
        gn = _clip(m)
        return pre, _grads(m), gn

    pre0, post0, gn0 = run(0.0)
    pre1, post1, gn1 = run(1e8)
    assert torch.equal(gn0, gn1)
    assert float(gn0) > 5.0                                        # the main group's clip binds
    for n in post0:
        if ".sign_block." in n:
            continue
        if n in SIGN_ROW:
            assert torch.equal(post0[n][[0, 2]], post1[n][[0, 2]]), n
        else:
            assert torch.equal(post0[n], post1[n]), n
    side = [post1[n] for n in post1 if ".sign_block." in n] + [post1[n][1] for n in SIGN_ROW]
    side_pre = [pre1[n] for n in pre1 if ".sign_block." in n] + [pre1[n][1] for n in SIGN_ROW]
    norm = lambda ts: float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(t) for t in ts])))
    assert norm(side_pre) > 1e3 and abs(norm(side) - 5.0) < 1e-4, (norm(side_pre), norm(side))
    # the main group's clipped norm is the max norm
    main = [post1[n] for n in post1 if ".sign_block." not in n and n not in SIGN_ROW] + \
           [post1[n][[0, 2]] for n in SIGN_ROW]
    assert abs(norm(main) - 5.0) < 1e-4
    # a global clip (today's) would have shrunk the trunk by the sign term
    m.load_state_dict(sd)
    m.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    out = forward(m, bt)
    (1e3 * (out[:, 0].square().sum() + out[:, 2].sum()) + 1e8 * out[:, 1].square().sum()).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
    k = "net.net.linear_in.weight"
    assert float(m.get_parameter(k).grad.norm()) < 1e-2 * float(post0[k].norm())


def test_clip_flag_off_is_torch_clip():
    bt = batch(seed=2)
    m, _ = build(branch=False)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    torch.manual_seed(0)
    forward(m, bt).square().sum().mul(1e3).backward()
    gn = _clip(m)
    g1 = _grads(m)
    m.load_state_dict(sd)
    m.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    forward(m, bt).square().sum().mul(1e3).backward()
    gt = torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0, error_if_nonfinite=False)
    assert torch.equal(gn, gt)
    for n, t in _grads(m).items():
        assert torch.equal(t, g1[n]), n


def test_sigma_stage_and_muon():
    bt = batch(seed=3)
    m, _ = build(branch=True)
    cfg = types.SimpleNamespace(model=types.SimpleNamespace(net=types.SimpleNamespace(out_channels=1)))
    E._freeze_all_but_sigma_head(types.SimpleNamespace(cfg=cfg, model=m))
    assert not any(p.requires_grad for p in m.net.net.sign_block.parameters())
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = MuonAdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(2):
        opt.zero_grad()
        out = forward(m, bt)
        (out[:, 2].sum() + out[:, 1].sum()).backward()
        _clip(m)
        opt.step()
    for n, p in m.named_parameters():
        if n in SIGN_ROW:
            assert torch.equal(p[:2], before[n][:2]) and not torch.equal(p[2], before[n][2]), n
        else:
            assert torch.equal(p, before[n]), n
    # in training, MuonAdamW puts sign_block's hidden matrices in its Muon groups
    m2, _ = build(branch=True)
    opt = MuonAdamW(list(m2.parameters()), lr=1e-3)
    muon_ids = {id(p) for g in opt.param_groups if g["use_muon"] for p in g["params"]}
    sb = [p for n, p in m2.net.net.sign_block.named_parameters() if p.dim() == 2]
    assert sb and all(id(p) in muon_ids for p in sb)


def test_assertions():
    with pytest.raises(AssertionError):
        build(sign=False, branch=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
