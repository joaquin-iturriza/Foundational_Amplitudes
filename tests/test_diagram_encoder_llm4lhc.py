"""CPU tests of the arXiv:2606.23791 diagram encoder port (models/diagram_encoder_llm4lhc.py).

Run: python tests/test_diagram_encoder_llm4lhc.py   (no GPU, no xformers; the wrapper
forward is exercised only where lloca/lgatr/mup import).
"""

import copy
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import models.diagram_encoder_llm4lhc as _m
except ImportError:  # models/__init__ pulls einops/lgatr/...; load the file on its own
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "diagram_encoder_llm4lhc", os.path.join(ROOT, "models", "diagram_encoder_llm4lhc.py"))
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
EXT_IN_TYPE, EXT_OUT_TYPE, VERTEX_TYPE = _m.EXT_IN_TYPE, _m.EXT_OUT_TYPE, _m.VERTEX_TYPE
LLM4LHCDiagramEncoder, graph_from_sidecar_diagram = _m.LLM4LHCDiagramEncoder, _m.graph_from_sidecar_diagram
inject_at_positions, laplacian_pe = _m.inject_at_positions, _m.laplacian_pe
load_llm4lhc_diagram_set, particle_label_vocab = _m.load_llm4lhc_diagram_set, _m.particle_label_vocab
process_token_add = _m.process_token_add

torch.set_default_dtype(torch.float64)   # tight comparisons against the reference


# ---------------------------------------------------------------------------------------
# synthetic sidecars in the tools/dump_diagrams.py schema
# ---------------------------------------------------------------------------------------
def _ext(n, pdg, state):
    return {"kind": "external", "interaction_id": 0, "leg_number": n, "pdg": pdg, "state": state}


def _vtx(orders):
    return {"kind": "vertex", "interaction_id": 1, "orders": orders}


def _e(u, v, pdg, ext, leg=0):
    return {"u": u, "v": v, "pdg": pdg, "external": ext, "leg_number": leg}


def schannel(prop):
    """e+ e- -> u u~ through an s-channel ``prop`` (MadGraph may sign it -22/-23).
    External nodes listed AFTER the vertices on purpose (reordering is tested)."""
    nodes = [_vtx({"QED": 1}), _vtx({"QED": 1}),
             _ext(1, -11, "in"), _ext(2, 11, "in"), _ext(3, 2, "out"), _ext(4, -2, "out")]
    # the external edge PDGs here are deliberately the crossed ones: the loader must take
    # the legs' identity from the subprocess's `external` list by leg number
    edges = [_e(2, 0, 11, True, 1), _e(3, 0, -11, True, 2), _e(1, 4, 2, True, 3),
             _e(1, 5, -2, True, 4), _e(0, 1, prop, False)]
    return {"nodes": nodes, "edges": edges, "diagram_type": "tree"}


EXT_EEUU = [{"number": 1, "pdg": -11, "state": "in"}, {"number": 2, "pdg": 11, "state": "in"},
            {"number": 3, "pdg": 2, "state": "out"}, {"number": 4, "pdg": -2, "state": "out"}]


def tchannel_qqg():
    """u u~ -> g g with t-channel u, s-channel g (two diagrams of different topology)."""
    ext = [{"number": 1, "pdg": 2, "state": "in"}, {"number": 2, "pdg": -2, "state": "in"},
           {"number": 3, "pdg": 21, "state": "out"}, {"number": 4, "pdg": 21, "state": "out"}]
    d1 = {"nodes": [_ext(1, 2, "in"), _ext(2, -2, "in"), _ext(3, 21, "out"), _ext(4, 21, "out"),
                    _vtx({"QCD": 1}), _vtx({"QCD": 1})],
          "edges": [_e(0, 4, 2, True, 1), _e(1, 5, -2, True, 2), _e(4, 2, 21, True, 3),
                    _e(5, 3, 21, True, 4), _e(4, 5, 2, False)]}
    d2 = {"nodes": [_ext(1, 2, "in"), _ext(2, -2, "in"), _ext(3, 21, "out"), _ext(4, 21, "out"),
                    _vtx({"QCD": 1}), _vtx({"QCD": 1})],
          "edges": [_e(0, 4, 2, True, 1), _e(1, 4, -2, True, 2), _e(5, 2, 21, True, 3),
                    _e(5, 3, 21, True, 4), _e(4, 5, -21, False)]}
    d3 = {"nodes": [_ext(1, 2, "in"), _ext(2, -2, "in"), _ext(3, 21, "out"), _ext(4, 21, "out"),
                    _vtx({"QCD": 1}), _vtx({"QCD": 1})],
          "edges": [_e(0, 4, 2, True, 1), _e(1, 5, -2, True, 2), _e(4, 3, 21, True, 4),
                    _e(5, 2, 21, True, 3), _e(4, 5, 2, False)]}
    return {"subprocesses": [{"external": ext, "diagrams": [d1, d2, d3]}]}


def bubble():
    """A one-loop-like graph: two propagators between the same two vertices (a multi-edge)
    plus a 4-leg external structure; N=7 nodes so the padding is exercised."""
    ext = [{"number": 1, "pdg": -11, "state": "in"}, {"number": 2, "pdg": 11, "state": "in"},
           {"number": 3, "pdg": 1, "state": "out"}, {"number": 4, "pdg": -1, "state": "out"}]
    d = {"nodes": [_ext(1, -11, "in"), _ext(2, 11, "in"), _ext(3, 1, "out"), _ext(4, -1, "out"),
                   _vtx({"QED": 1}), _vtx({"QED": 1}), _vtx({"QCD": 1})],
         "edges": [_e(0, 4, -11, True, 1), _e(1, 4, 11, True, 2), _e(4, 5, 23, False),
                   _e(5, 6, 1, False), _e(5, 6, -1, False), _e(6, 2, 1, True, 3),
                   _e(6, 3, -1, True, 4)],
         "diagram_type": "loop"}
    return {"subprocesses": [{"external": ext, "diagrams": [d]}]}


def payloads():
    p0 = {"subprocesses": [{"external": EXT_EEUU, "diagrams": [schannel(-22), schannel(-23)]}]}
    # flavour-summed: two subprocesses, all diagrams pooled
    ext_dd = [dict(l, pdg={2: 1, -2: -1}.get(l["pdg"], l["pdg"])) for l in EXT_EEUU]
    p1 = {"subprocesses": [{"external": EXT_EEUU, "diagrams": [schannel(22)]},
                           {"external": ext_dd, "diagrams": [schannel(23)]}]}
    return [p0, p1, tchannel_qqg(), bubble()]


def build(mode="pooled", efm="bias_kv", seed=0, out_dim=24, pl=None, randomize_kv=True):
    ds = load_llm4lhc_diagram_set(pl if pl is not None else payloads(), laplacian_pe_dim=4)
    torch.manual_seed(seed)
    enc = LLM4LHCDiagramEncoder(ds, out_dim=out_dim, hidden_size=16, num_layers=2,
                                num_heads=4, laplacian_pe_dim=4, mode=mode,
                                edge_feature_mode=efm)
    if randomize_kv and efm == "bias_kv" and mode != "pooled_external":
        with torch.no_grad():   # zero-init tables would hide the K/V terms in the checks
            enc.edge_key_embed.weight.normal_(0, 0.5)
            enc.edge_val_embed.weight.normal_(0, 0.5)
    return enc


# ---------------------------------------------------------------------------------------
# the reference: a direct transcription of their DiagramEncoder (padded (P, D_max, N),
# materialised per-pair edge key/value tensors, masked softmax over D)
# ---------------------------------------------------------------------------------------
def reference_pooled(enc):
    P, H, h = enc.n_proc, enc.hidden_size, enc.num_heads
    hd = H // h
    seg = enc.seg.tolist()
    D = max(seg.count(p) for p in range(P))
    N = enc.node_types.shape[1]
    idx = torch.full((P, D), -1, dtype=torch.long)
    cnt = [0] * P
    for t, p in enumerate(seg):
        idx[p, cnt[p]] = t
        cnt[p] += 1
    dmask = idx >= 0
    src = idx.clamp(min=0)

    def pad(x):
        y = x[src.flatten()].reshape(P, D, *x.shape[1:]).clone()
        y[~dmask] = 0
        return y
    types, nmask = pad(enc.node_types), pad(enc.node_mask)
    adj, labels, pe = pad(enc.adj), pad(enc.edge_labels), pad(enc.lap_pe)
    B, M = P * D, N + 1
    x = enc.node_type_embed(types) + enc.pe_proj(pe.to(enc.pe_proj.weight.dtype))
    x = (x * nmask.unsqueeze(-1)).reshape(B, N, H)
    x = torch.cat([enc.cls_token.expand(B, 1, H), x], 1)
    am = torch.cat([torch.ones(B, 1, dtype=torch.bool), nmask.reshape(B, N)], 1)
    eb = (enc.edge_label_bias(labels) * adj.unsqueeze(-1)).permute(0, 1, 4, 2, 3).reshape(B, h, N, N)
    edge_bias = x.new_zeros(B, h, M, M)
    edge_bias[:, :, 1:, 1:] = eb

    def kv(emb):
        f = (emb(labels) * adj.unsqueeze(-1)).reshape(P, D, N, N, h, hd)
        f = f.permute(0, 1, 4, 2, 3, 5).reshape(B, h, N, N, hd)
        full = x.new_zeros(B, h, M, M, hd)
        full[:, :, 1:, 1:] = f
        return full
    ek = kv(enc.edge_key_embed) if enc.edge_feature_mode == "bias_kv" else None
    ev = kv(enc.edge_val_embed) if enc.edge_feature_mode == "bias_kv" else None
    for L in enc.layers:
        scale = 1 / math.sqrt(hd)
        qkv = L.qkv(L.norm1(x)).reshape(B, M, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        s = (q @ k.transpose(-2, -1)) * scale + edge_bias
        if ek is not None:
            s = s + torch.einsum("bhmd,bhmnd->bhmn", q, ek) * scale
        s = s.masked_fill((~am)[:, None, None, :], float("-inf"))
        a = s.softmax(-1)
        o = a @ v
        if ev is not None:
            o = o + torch.einsum("bhmn,bhmnd->bhmd", a, ev)
        x = x + L.out(o.transpose(1, 2).reshape(B, M, H))
        x = x + L.ffn(L.norm2(x))
    per = enc.out_proj(enc.norm_out(x[:, 0])).reshape(P, D, -1)
    lg = (per * enc.pool_query).sum(-1) / math.sqrt(enc.out_dim)
    w = lg.masked_fill(~dmask, float("-inf")).softmax(-1).unsqueeze(-1)
    return (per * w).sum(1), per, dmask


# ---------------------------------------------------------------------------------------
def test_graph_construction():
    vocab = particle_label_vocab()
    # their 30 SM labels first, then the fixed one-loop ghosts and Goldstones (their data has no loops)
    loop = _m._LOOP_LINES
    n = 30 + len(loop)
    assert len(vocab) == n and sorted(vocab.values()) == list(range(n))
    assert all(vocab[p] >= 30 for p in loop) and all(vocab[p] < 30 for p in vocab if p not in loop)
    assert _m._label(-250, vocab) == vocab[250] and _m._label(-82, vocab) != _m._label(82, vocab)
    nt, edges = graph_from_sidecar_diagram(schannel(-22), EXT_EEUU, vocab)
    # externals first, by leg number, typed in/out; then the vertices
    assert nt == [EXT_IN_TYPE, EXT_IN_TYPE, EXT_OUT_TYPE, EXT_OUT_TYPE, VERTEX_TYPE, VERTEX_TYPE]
    lab = {(min(i, j), max(i, j)): l for i, j, l in edges}
    # external legs carry the process definition's identity (e+ on leg 1), not the
    # crossed edge PDG; the -22 propagator is folded to the photon
    assert lab[(0, 4)] == vocab[-11] and lab[(1, 4)] == vocab[11]
    assert lab[(4, 5)] == vocab[22]
    ds = load_llm4lhc_diagram_set(payloads(), laplacian_pe_dim=4)
    assert ds.n_total == 2 + 2 + 3 + 1 and ds.n_diagrams.tolist() == [2, 2, 3, 1]
    assert ds.node_types.shape == (8, 7) and ds.lap_pe.shape == (8, 7, 4)
    assert torch.equal(ds.adj, ds.adj.transpose(1, 2))
    assert ds.ext_count.tolist() == [4, 4, 4, 4]
    assert ds.ext_dir[0].tolist() == [1, 1, 2, 2]
    # pad nodes carry nothing
    assert not ds.adj[~ds.node_mask].any() and (ds.node_types[~ds.node_mask] == 0).all()
    # an unknown particle fails loudly
    bad = {"subprocesses": [{"external": EXT_EEUU, "diagrams": [schannel(9000099)]}]}   # not a ghost or Goldstone
    try:
        load_llm4lhc_diagram_set([bad])
        raise AssertionError("unknown PDG accepted")
    except KeyError:
        pass
    print("graph construction: ok")


def test_laplacian_pe():
    # path graph 0-1-2-3: eigenvectors of the normalised Laplacian, sign-fixed
    a = np.zeros((4, 4), bool)
    for i in range(3):
        a[i, i + 1] = a[i + 1, i] = True
    pe = laplacian_pe(a, 6)
    assert pe.shape == (4, 6) and np.allclose(pe[:, 3:], 0)
    deg = a.sum(1)
    lap = np.eye(4) - a / np.sqrt(np.outer(deg, deg))
    for c in range(3):
        v = pe[:, c]
        lam = v @ lap @ v
        assert np.allclose(lap @ v, lam * v, atol=1e-5)
        assert v[np.argmax(np.abs(v))] >= 0
    print("laplacian PE: ok")


def test_shapes_and_modes():
    for mode, n in (("pooled", 4), ("per_diagram", 8), ("pooled_external", 4)):
        for efm in ("bias", "bias_kv"):
            if mode == "pooled_external" and efm == "bias_kv":
                continue
            out = build(mode=mode, efm=efm)()
            assert out.shape == (n, 24) and torch.isfinite(out).all(), (mode, efm, out.shape)
    print("shapes (pooled, per_diagram, pooled_external x bias, bias_kv): ok")


def test_matches_reference():
    for efm in ("bias", "bias_kv"):
        enc = build(efm=efm)
        enc.eval()
        ours = enc()
        ref, per_ref, dmask = reference_pooled(enc)
        err = (ours - ref).abs().max().item()
        assert err < 1e-10, (efm, err)
        per = enc.encode_diagrams()
        seg = enc.seg.tolist()
        k = [0] * enc.n_proc
        for t, p in enumerate(seg):
            assert torch.allclose(per[t], per_ref[p, k[p]], atol=1e-10)
            k[p] += 1
        print(f"matches a direct transcription of their encoder ({efm}): max |diff| {err:.1e}")


def test_determinism():
    a, b = build(seed=3), build(seed=3)
    assert torch.equal(a(), b()) and torch.equal(a(), a())
    c = build(seed=4)
    assert not torch.allclose(a(), c())
    # zero-init K/V tables do not perturb the RNG: the shared parameters of a "bias" and
    # a "bias_kv" encoder built from one seed are identical (their snapshot/restore)
    x = build(efm="bias", seed=5, randomize_kv=False)
    y = build(efm="bias_kv", seed=5, randomize_kv=False)
    sx, sy = x.state_dict(), y.state_dict()
    assert all(torch.equal(sx[k], sy[k]) for k in sx)
    # ...and with the tables at zero the two compute the same function
    assert torch.allclose(x(), y(), atol=1e-12)
    print("determinism (seeded rebuild, repeat call, bias/bias_kv RNG isolation): ok")


def test_diagram_permutation():
    pl = payloads()
    enc = build(pl=pl)
    e0 = enc()
    pl2 = copy.deepcopy(pl)
    pl2[2]["subprocesses"][0]["diagrams"].reverse()                    # within a process
    pl2[1]["subprocesses"].reverse()                                  # across subprocesses
    enc2 = build(pl=pl2)
    enc2.load_state_dict(enc.state_dict())
    e2 = enc2()
    assert torch.allclose(e0, e2, atol=1e-10)
    # per_diagram is equivariant instead: the readouts are permuted with the diagrams
    pd = build(mode="per_diagram", pl=pl)
    pd2 = build(mode="per_diagram", pl=pl2)
    pd2.load_state_dict(pd.state_dict())
    r, r2 = pd(), pd2()
    assert torch.allclose(r[4:7], r2[4:7].flip(0), atol=1e-10)
    # the order the sidecar lists the external nodes in does not matter (re-sorted by leg
    # number on load); vertices keep the sidecar order, as their converter's vtx index
    # (a vertex relabelling changes the LapPE signs on symmetric graphs, theirs too)
    pl3 = copy.deepcopy(pl)
    g = pl3[0]["subprocesses"][0]["diagrams"][0]
    perm = [5, 0, 3, 2, 1, 4]
    inv = {old: new for new, old in enumerate(perm)}
    g["nodes"] = [g["nodes"][i] for i in perm]
    for e in g["edges"]:
        e["u"], e["v"] = inv[e["u"]], inv[e["v"]]
    enc3 = build(pl=pl3)
    enc3.load_state_dict(enc.state_dict())
    assert torch.allclose(e0, enc3(), atol=1e-10)
    print("diagram permutation: pooled invariant, per_diagram equivariant, external node order irrelevant: ok")


def test_injection_and_gradients():
    enc = build()
    E = enc()                                                          # (4, 24)
    ptr = torch.tensor([0, 3, 7, 9, 13])
    pids = torch.tensor([2, 0, 2, 3])
    add = process_token_add(E, pids, ptr, 13, 24)
    want = torch.cat([E[p].expand(int(ptr[i + 1] - ptr[i]), -1) for i, p in enumerate(pids.tolist())])
    assert torch.equal(add, want)
    # their index_put(accumulate=True): repeated positions sum, others untouched
    base = torch.ones(5, 3)
    got = inject_at_positions(base, torch.full((3, 3), 2.0), (torch.tensor([1, 1, 4]),))
    assert got[1].eq(5).all() and got[4].eq(3).all() and got[0].eq(1).all()
    # gradient reaches every encoder parameter through the token term
    h = torch.randn(13, 24)
    loss = ((h + add) ** 2).sum()
    loss.backward()
    dead = [n for n, p in enc.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, dead
    # process 1 is absent from the batch: its diagrams get no gradient through the pool,
    # but the shared weights do; a zero-init K/V table also receives gradient
    enc0 = build(randomize_kv=False)
    process_token_add(enc0(), pids, ptr, 13, 24).sum().backward()
    assert enc0.edge_key_embed.weight.grad.abs().sum() > 0
    assert enc0.edge_val_embed.weight.grad.abs().sum() > 0
    print("injection (index_put accumulate, per-particle broadcast) and gradient flow: ok")


def test_real_sidecar():
    path = os.path.join(os.path.dirname(ROOT), "data", "diagrams", "syn_00000.diagrams.json")
    for cand in (path, "/home/joaquin/work/FA/data/diagrams/syn_00000.diagrams.json"):
        if os.path.exists(cand):
            ds = load_llm4lhc_diagram_set([cand], laplacian_pe_dim=8)
            torch.manual_seed(0)
            enc = LLM4LHCDiagramEncoder(ds, out_dim=128)
            out = enc()
            assert out.shape == (1, 128) and torch.isfinite(out).all()
            n = sum(p.numel() for p in enc.parameters())
            print(f"real sidecar {os.path.basename(cand)}: D={ds.n_total}, out {tuple(out.shape)}, "
                  f"{n} parameters at the paper's size (256 wide, 3 layers, 4 heads)")
            return
    print("real sidecar: none found, skipped")


def test_wrapper_forward():
    try:
        from wrappers import AmplitudeLLoCaWrapper  # noqa: F401  (needs lgatr, lloca, mup)
        from models.lloca import LLOCAMuPTransformer
    except Exception as exc:  # the laptop driver env has none of them
        print(f"wrapper forward: skipped ({type(exc).__name__}: {exc})")
        return
    from wrappers import AmplitudeLLoCaWrapper
    torch.set_default_dtype(torch.float32)
    net = LLOCAMuPTransformer(num_scalars=4, hidden_channels_mlp=16, num_layers_mlp=2,
                              in_channels=8, attn_reps="8x0n+2x1n", out_channels=1,
                              num_blocks=2, num_heads=2)
    w = AmplitudeLLoCaWrapper(net, token_size=0)
    ds = load_llm4lhc_diagram_set(payloads(), laplacian_pe_dim=8)
    enc = LLM4LHCDiagramEncoder(ds, out_dim=net.net.hidden_channels)
    w.setup_diagram_conditioning_llm4lhc(enc)
    w.use_pids, w.particle_encoder = True, None
    w.token_size = 4
    ptr = torch.tensor([0, 4, 8])
    p = torch.randn(8, 4).abs() + 1
    out = w(p, torch.zeros(8, dtype=torch.long), 0.0, 1.0, ptr,
            process_ids=torch.tensor([0, 2]))
    assert out.shape == (2, 1) and torch.isfinite(out).all()
    print("wrapper forward on CPU: ok")
    torch.set_default_dtype(torch.float64)


if __name__ == "__main__":
    test_graph_construction()
    test_laplacian_pe()
    test_shapes_and_modes()
    test_matches_reference()
    test_determinism()
    test_diagram_permutation()
    test_injection_and_gradients()
    test_real_sidecar()
    test_wrapper_forward()
    print("all passed")
