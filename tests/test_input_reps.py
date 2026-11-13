#!/usr/bin/env python
"""The three input representations for singular limits (D374, kit_benchmark input arms; config/model/lloca.yaml):
model.input_pair_logs, model.input_soft_eps, model.input_local_log_eps. For each flag: the μP LLoCa wrapper's output is
invariant under a random Lorentz transform of every momentum (float64), and the flag changes the output (it is wired).
Plus the pair logs' values on a known event and the soft rescale's fixed point x = 1.

Run:  .venv-cpu/bin/python tests/test_input_reps.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wrappers import AmplitudeLLoCaWrapper, _asym_mu, _asym_terms, _div_form_inputs, _div_form_width, _pair_log_inputs, _soft_rescale  # noqa: E402

torch.set_default_dtype(torch.float64)
LENS = (5, 5, 4)


def _events(seed, lens=LENS):
    """Events of `lens` particles: two back-to-back massless beams, then massless outgoing momenta, one of them soft."""
    rng = np.random.default_rng(seed)
    out = []
    for n in lens:
        p3 = rng.normal(size=(n - 2, 3))
        p3[-1] *= 1e-6                                    # a soft leg
        ev = [[1, 0, 0, 1], [1, 0, 0, -1]] + [[np.linalg.norm(v), *v] for v in p3]
        out.append(ev)
    return torch.tensor(np.concatenate(out)), torch.tensor(np.cumsum([0, *lens]))


def _lorentz(seed):
    rng = np.random.default_rng(seed)
    b = rng.normal(size=3)
    b *= 0.7 / np.linalg.norm(b)
    g = 1 / np.sqrt(1 - b @ b)
    B = np.eye(4)
    B[0, 0] = g
    B[0, 1:] = B[1:, 0] = g * b
    B[1:, 1:] += (g - 1) * np.outer(b, b) / (b @ b)
    R = np.eye(4)
    R[1:, 1:] = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    return torch.tensor(B @ R)


def _wrapper(**flags):
    from models.lloca import LLOCAMuPTransformer
    n_pl = flags.get("input_pair_logs", 0)
    ns = 16 + n_pl * (n_pl - 1) // 2 + (_div_form_width(n_pl) if flags.get("input_div_form") else 0)
    torch.manual_seed(0)
    net = LLOCAMuPTransformer(num_scalars=ns, hidden_channels_mlp=16, num_layers_mlp=2, in_channels=ns + 4,
                              attn_reps="8x0n+2x1n", out_channels=1, num_blocks=2, num_heads=2)
    w = AmplitudeLLoCaWrapper(net, token_size=0, d_particle_hidden=16, **flags)
    w.setup_particle_features(use_pids=False, property_matrix=torch.randn(5, 4).numpy())
    return w.double().eval()


def _out(w, p, ptr, kinds=(1, 1, 2, 3, 4, 2)):
    tok = torch.cat([torch.tensor(kinds)[:n] for n in (ptr[1:] - ptr[:-1]).tolist()])
    with torch.no_grad():
        return w(p, tok, 0.0, 1.0, ptr).reshape(-1)


FLAGS = {"pair_logs": dict(input_pair_logs=5), "soft_rescale": dict(input_soft_eps=1e-8),
         "local_logs": dict(input_local_log_eps=1e-8),
         "both_6": dict(input_pair_logs=6, input_soft_eps=1e-8)}     # the uugg flags: 15 pair logs + soft rescale


def _check(name, lens=LENS):
    p, ptr = _events(1, lens)
    L = _lorentz(2)
    w = _wrapper(**FLAGS[name])
    a, b = _out(w, p, ptr), _out(w, p @ L.T, ptr)
    rel = float((a - b).abs().max() / a.abs().max())
    # float64 frame rounding (~1e-16) is amplified near z = 0 by up to 1/(eps asinh(1/eps)) ~ 5e6 under local_logs
    assert rel < 1e-7, f"{name}: not Lorentz invariant, rel {rel:.2e}"
    moved = float((a - _out(_wrapper(), p, ptr)).abs().max() / a.abs().max())
    assert moved > 1e-6, f"{name}: the flag does not change the output"
    print(f"{name}: invariant (rel {rel:.1e}), flag moves the output (rel {moved:.1e}): ok")


def test_pair_logs_equivariant():
    _check("pair_logs")


def test_soft_rescale_equivariant():
    _check("soft_rescale")


def test_local_logs_equivariant():
    _check("local_logs")


def test_both_flags_6particle_equivariant():
    _check("both_6", lens=(6, 5, 4))


def test_pair_log_values():
    p, ptr = _events(3)
    f = _pair_log_inputs(p, ptr, 5)
    assert f.shape == (14, 10)
    i, j, k = 2, 4, 0                                  # event 0: pair (2, 4) is column 7 of the i<j order
    col = [(a, c) for a in range(5) for c in range(a + 1, 5)].index((i, j))
    dot = p[i, 0] * p[j, 0] - (p[i, 1:] * p[j, 1:]).sum()
    assert torch.allclose(f[k, col], torch.log(2 * dot / 4.0) / 5)
    assert (f[10:, [3, 6, 8, 9]] == 0).all()           # the 4-particle event's pairs with leg 4 are 0
    assert (f[0] == f[4]).all()                        # broadcast over the event


def test_pair_threshold_log_beta():
    """input_pair_threshold: a t t~ pair at sqrt s gives log(beta^2)/5, beta^2 = 1 - 4 m^2/s; massless pairs unchanged."""
    m, sq = 173.0, 346.5
    E, pz = sq / 2, (sq * sq / 4 - m * m) ** 0.5
    p = torch.tensor([[E, 0, 0, E], [E, 0, 0, -E], [E, pz * 0.6, 0, pz * 0.8], [E, -pz * 0.6, 0, -pz * 0.8]],
                     dtype=torch.float64)
    ptr = torch.tensor([0, 4])
    f, g = _pair_log_inputs(p, ptr, 4, threshold=True), _pair_log_inputs(p, ptr, 4)
    beta2 = 1 - 4 * m * m / sq ** 2
    assert torch.allclose(f[0, 5], torch.log(torch.tensor(beta2, dtype=torch.float64)) / 5)   # pair (2, 3)
    assert torch.allclose(f[0, :5], g[0, :5])                                                  # pairs with a massless leg


def test_scalars_only_reads_no_momentum():
    """input_scalars_only: with the pair logs held fixed, two different event sets give the same output (no momentum
    reaches the backbone but through the pair logs), and the flag changes the output against the pair logs alone."""
    import wrappers
    p, ptr = _events(1)
    q, _ = _events(5)
    fixed = _pair_log_inputs(p, ptr, 5)
    w = _wrapper(input_pair_logs=5, input_scalars_only=True)
    orig = wrappers._pair_log_inputs
    wrappers._pair_log_inputs = lambda *a, **k: fixed
    try:
        a, b = _out(w, p, ptr), _out(w, q, ptr)
    finally:
        wrappers._pair_log_inputs = orig
    assert torch.equal(a, b), f"scalars_only reads the momenta: {float((a - b).abs().max()):.2e}"
    moved = float((a - _out(_wrapper(input_pair_logs=5), p, ptr)).abs().max() / a.abs().max())
    assert moved > 1e-6, "input_scalars_only does not change the output"


def test_route_off():
    """input_route_off: each cut keeps the output Lorentz invariant and changes it against the pair logs alone; with
    'local' the momenta still reach the backbone through the learned frames (unlike input_scalars_only)."""
    import wrappers
    p, ptr = _events(1)
    L = _lorentz(2)
    base = _out(_wrapper(input_pair_logs=5), p, ptr)
    for r in ("local", "frames"):
        w = _wrapper(input_pair_logs=5, input_route_off=r)
        a, b = _out(w, p, ptr), _out(w, p @ L.T, ptr)
        assert float((a - b).abs().max() / a.abs().max()) < 1e-7, f"route_off={r}: not Lorentz invariant"
        assert float((a - base).abs().max() / a.abs().max()) > 1e-6, f"route_off={r} does not change the output"
    q, _ = _events(5)
    fixed = _pair_log_inputs(p, ptr, 5)
    w = _wrapper(input_pair_logs=5, input_route_off="local")
    orig = wrappers._pair_log_inputs
    wrappers._pair_log_inputs = lambda *a, **k: fixed
    try:
        a, b = _out(w, p, ptr), _out(w, q, ptr)
    finally:
        wrappers._pair_log_inputs = orig
    assert not torch.equal(a, b), "route_off=local: the frames carry no momentum"


def test_log_skip():
    """input_log_skip: Lorentz invariant, changes the output, and equals the sum written out pair by pair."""
    import wrappers
    p, ptr = _events(1)
    L = _lorentz(2)
    for thr in (False, True):
        w = _wrapper(input_pair_logs=5, input_log_skip=2, input_pair_threshold=thr)
        a, b = _out(w, p, ptr), _out(w, p @ L.T, ptr)
        assert float((a - b).abs().max() / a.abs().max()) < 1e-7, "log skip: not Lorentz invariant"
    w0, w = _wrapper(input_pair_logs=5), _wrapper(input_pair_logs=5, input_log_skip=2)
    assert float((_out(w, p, ptr) - _out(w0, p, ptr)).abs().max()) > 1e-6, "log skip does not change the output"
    types = torch.randn(p.shape[0], 16, dtype=torch.float64)
    head = torch.nn.Linear(16, 6).double()
    eta = torch.tensor([1.0, -1, -1, -1], dtype=torch.float64)
    h = head(types).detach()
    for thr, e in ((t, e) for t in (False, True) for e in range(len(ptr) - 1)):
        got = wrappers._log_skip(p, ptr, types, 5, head, thr)
        q, he = p[ptr[e]:ptr[e + 1]].double(), h[ptr[e]:ptr[e + 1]]
        s = float(((q[0] + q[1]) ** 2 * eta).sum())
        want = 0.0
        for i in range(len(q)):
            m2 = float((q[i] ** 2 * eta).sum())
            Mi = np.log(max(abs(m2) / s, 1e-16)) / 5 if abs(m2) / s > 1e-12 else 0.0
            want += float(he[i, 4]) * Mi + float(he[i, 5]) * Mi ** 2
            for j in range(i + 1, len(q)):
                mm = np.sqrt(abs(float((q[i] ** 2 * eta).sum())) * abs(float((q[j] ** 2 * eta).sum()))) if thr else 0.0
                Lij = np.log(max(abs(2 * (float((q[i] * q[j] * eta).sum()) - mm) / s), 1e-16)) / 5
                want += float(he[i, :2] @ he[j, :2]) * Lij + float(he[i, 2:4] @ he[j, 2:4]) * Lij ** 2
        assert abs(float(got[e]) - want) < 1e-9 * max(1.0, abs(want)), (e, float(got[e]), want)
    # linear only: head columns of the quadratic terms (2:4 pair, 5 mass) must not matter
    head2 = torch.nn.Linear(16, 6).double()
    with torch.no_grad():
        head2.weight[[0, 1, 4]] = head.weight[[0, 1, 4]]
        head2.bias[[0, 1, 4]] = head.bias[[0, 1, 4]]
    for thr in (False, True):
        a = wrappers._log_skip(p, ptr, types, 5, head, thr, quadratic=False)
        b = wrappers._log_skip(p, ptr, types, 5, head2, thr, quadratic=False)
        assert torch.allclose(a, b, atol=1e-12), "linear-only skip reads the quadratic coefficients"
        assert not torch.allclose(a, wrappers._log_skip(p, ptr, types, 5, head, thr)), "linear-only equals the full skip"
    wl = _wrapper(input_pair_logs=5, input_log_skip=2, input_log_skip_linear=True)
    a, b = _out(wl, p, ptr), _out(wl, p @ L.T, ptr)
    assert float((a - b).abs().max() / a.abs().max()) < 1e-7, "linear-only log skip: not Lorentz invariant"


def test_soft_rescale_fixed_point():
    p, ptr = _events(4)
    q = _soft_rescale(p, ptr, 1e-8)
    assert torch.allclose(q[:2], p[:2])                # the beams have x = 1
    assert float(q[4, 0] / p[4, 0]) > 1e5              # the soft leg is lifted (x ~ 1e-6: log1p(100)/(1e-6 log 1e8))


def test_div_form():
    """input_div_form: values of each block against the formulas written out, Lorentz invariance of the features and
    of the wrapper's output, and the flag moves the output. Event 0: e+ e- > u u~ g g (soft last gluon), event 1:
    e+ e- > t t~ near threshold."""
    from particle_ids import GLOBAL_PROPERTY_MATRIX as G, global_encode
    eta = torch.tensor([1.0, -1, -1, -1], dtype=torch.float64)
    dot = lambda x, y: float((x * y * eta).sum())
    p0, _ = _events(5, (6,))
    m, sq = 172.5, 2 * 172.5 * 1.0001
    E, pz = sq / 2, (sq * sq / 4 - m * m) ** 0.5
    tt = torch.tensor([[E, 0, 0, E], [E, 0, 0, -E], [E, pz * 0.6, 0, pz * 0.8], [E, -pz * 0.6, 0, -pz * 0.8]]) / 100
    p = torch.cat([p0, tt]); ptr = torch.tensor([0, 6, 10])
    tok = torch.tensor(np.concatenate([global_encode(np.array([-11, 11, 2, -2, 21, 21])),
                                       global_encode(np.array([-11, 11, 6, -6]))]))
    Gt = torch.as_tensor(G)
    cas, heavy = Gt[tok, 7], Gt[tok, 2] >= 0
    f = _div_form_inputs(p, ptr, 6, cas, heavy)
    assert f.shape == (10, _div_form_width(6))
    tri = [(i, j, k) for i in range(6) for k in range(i + 1, 6) for j in range(6) if j not in (i, k)]
    prs = [(a, b) for a in range(6) for b in range(a + 1, 6)]
    nt, npr = len(tri), len(prs)
    q = p[:6]; s = dot(q[0] + q[1], q[0] + q[1])
    # soft: (u=2, g=5, u~=3) is the eikonal of the soft gluon between the quarks; (e+=0, g, u) is 0 (e+ colourless)
    c = tri.index((2, 5, 3))
    want = np.log(s * 2 * dot(q[2], q[3]) / (2 * dot(q[2], q[5]) * 2 * dot(q[5], q[3]))) / 5
    assert abs(float(f[0, c]) - want) < 1e-9, (float(f[0, c]), want)
    assert float(f[0, tri.index((0, 5, 2))]) == 0 and float(f[0, tri.index((2, 3, 4))]) == 0   # j = u~ is no gluon
    # collinear: (u=2, g=4) q g, (g=4, g=5) g g, (u, u~) q q; beams 0
    Q = q[0] + q[1]
    z = dot(q[2], Q) / (dot(q[2], Q) + dot(q[4], Q))
    assert abs(float(f[1, nt + prs.index((2, 4))]) - np.log(4 / 3 * (1 + z * z) / (1 - z)) / 5) < 1e-9
    z = dot(q[4], Q) / (dot(q[4], Q) + dot(q[5], Q))
    assert abs(float(f[1, nt + prs.index((4, 5))]) - np.log(3 * (z / (1 - z) + (1 - z) / z + z * (1 - z))) / 5) < 1e-9
    z = dot(q[2], Q) / (dot(q[2], Q) + dot(q[3], Q))
    assert abs(float(f[1, nt + prs.index((2, 3))]) - np.log(0.5 * (z * z + (1 - z) ** 2)) / 5) < 1e-9
    assert (f[0, nt:nt + npr][[prs.index((0, 2)), prs.index((0, 1))]] == 0).all()
    assert (f[0, nt + npr:] == 0).all()                         # no heavy leg in the u u~ g g event
    # t t~: Coulomb log(1/beta)/5 with beta = sqrt(1 - 4 m^2/s); Sudakov log^2(s/m^2)/25 on the tops; no colour blocks
    beta = np.sqrt(1 - 4 * m * m / sq ** 2)
    assert abs(float(f[6, nt + npr + prs.index((2, 3))]) - np.log(1 / beta) / 5) < 1e-6
    assert abs(float(f[6, nt + npr + npr + 2]) - np.log(sq ** 2 / m ** 2) ** 2 / 25) < 1e-9
    assert (f[6, :nt + npr] == 0).all() and (f[6, nt + npr + npr + 4:] == 0).all()   # columns past 4 legs
    assert (f[6] == f[9]).all()
    L = _lorentz(2)
    assert torch.allclose(f, _div_form_inputs(p @ L.T, ptr, 6, cas, heavy), atol=1e-9, rtol=1e-9), "not Lorentz invariant"
    # through the wrapper (its property table indexed by the same global tokens)
    def wrap(**fl):
        from models.lloca import LLOCAMuPTransformer
        ns = 16 + 15 + (_div_form_width(6) if fl else 0)
        torch.manual_seed(0)
        net = LLOCAMuPTransformer(num_scalars=ns, hidden_channels_mlp=16, num_layers_mlp=2, in_channels=ns + 4,
                                  attn_reps="8x0n+2x1n", out_channels=1, num_blocks=2, num_heads=2)
        w = AmplitudeLLoCaWrapper(net, token_size=0, d_particle_hidden=16, input_pair_logs=6, **fl)
        w.setup_particle_features(use_pids=False, property_matrix=G)
        return w.double().eval()
    def out(w, x):
        with torch.no_grad():
            return w(x, tok, 0.0, 1.0, ptr).reshape(-1)
    w = wrap(input_div_form=True)
    a, b = out(w, p), out(w, p @ L.T)
    assert float((a - b).abs().max() / a.abs().max()) < 1e-7, "div form wrapper: not Lorentz invariant"
    assert float((a - out(wrap(), p)).abs().max() / a.abs().max()) > 1e-6, "div form does not change the output"


def _uug(x1, x2):
    """e+ e- > u u~ g at sqrt(s) = 1 from the energy fractions x1 (u), x2 (u~), x3 = 2 - x1 - x2 (g); s_ij = 1 - x_k."""
    E1, E2, E3 = x1 / 2, x2 / 2, 1 - (x1 + x2) / 2
    c = 1 - (1 - (2 - x1 - x2)) / (2 * E1 * E2)                       # cos of the u, u~ angle from s_12 = 1 - x3
    p1 = np.array([E1, E1 * np.sqrt(max(1 - c * c, 0)), 0, -E1 * c])
    p2 = np.array([E2, 0, 0, -E2])
    p3 = np.array([E3, *(-(p1[1:] + p2[1:]))])
    R = np.eye(4)
    R[1:, 1:] = [[np.cos(1.0), 0, np.sin(1.0)], [0, 1, 0], [-np.sin(1.0), 0, np.cos(1.0)]]   # no leg along a beam
    return np.array([[0.5, 0, 0, 0.5], [0.5, 0, 0, -0.5], R @ p1, R @ p2, R @ p3])


def _tt(beta, m=0.1725):
    """e+ e- > t t~ at the pair velocity beta, tops at 60 degrees to the beam."""
    E = m / np.sqrt(1 - beta ** 2)
    q = E * beta
    return np.array([[E, 0, 0, E], [E, 0, 0, -E], [E, q * 3 ** 0.5 / 2, 0, q / 2], [E, -q * 3 ** 0.5 / 2, 0, -q / 2]])


def test_asym_factor():
    """model.input_asym_factor: (1) the factors' limits written out: the q g dipole times s_qg/s goes to C_F (1+z^2)/(1-z)
    (z the quark's energy fraction of the pair) as u || g, the two dipoles' sum goes to the eikonal 2 C_F s s_qq~/(s_qg
    s_q~g) as the gluon goes soft, 1/beta and log^2(s/m^2) on a loop t t~ event and no term on a tree one; (2) exactness:
    at fixed readouts log|M|^2 - log(exp(c_t) D_t) of the dominant term goes to 0 (g -> 0), at y = s_qg/s, x = E_g/E down
    to 1e-7 and beta to 1e-6; (3) through the wrapper, MSE and HETEROSC: Lorentz invariance, the flag moves mu, sigma
    stays the last row, and on plain momenta with the frames' boosts capped mu's slope in log y is -1, in log x -2."""
    from particle_ids import GLOBAL_PROPERTY_MATRIX as G, global_encode
    Gt = torch.as_tensor(G)
    tok_uug = torch.tensor(global_encode(np.array([-11, 11, 2, -2, 21])))
    tok_tt = torch.tensor(global_encode(np.array([-11, 11, 6, -6])))

    def terms(ev, tok, loops):
        p = torch.tensor(ev)
        return _asym_terms(p, torch.tensor([0, len(ev)]), len(ev), Gt[tok, 7], Gt[tok, 2] >= 0,
                           torch.tensor([float(loops)]))

    def dip(ld, on, mem, i, j, k):
        t = [n for n in range(mem.shape[0]) if tuple(mem[n].tolist()) == (i, j, k)][0]
        assert bool(on[0, t]), (i, j, k)
        return float(ld[0, t].exp())

    # which terms exist on u u~ g: (u, g; u~) and (u~, g; u): two dipoles; no Coulomb, no Sudakov even at one loop
    ld, on, mem = terms(_uug(0.9, 0.8), tok_uug, 1)
    assert int(on.sum()) == 2, mem[on[0]]
    # collinear u || g: s_13 = 1 - x2 -> 0 at fixed x1 + x3; z = E_u/(E_u + E_g)
    for y in (1e-3, 1e-5, 1e-7):
        ev = _uug(0.7 + y * 0.3, 1 - y)
        ld, on, mem = terms(ev, tok_uug, 0)
        z = ev[2, 0] / (ev[2, 0] + ev[4, 0])
        r = dip(ld, on, mem, 2, 4, 3) * y / (4 / 3 * (1 + z * z) / (1 - z)) - 1
        assert abs(r) < 3 * y + 1e-6, (y, r)
    # soft g: x3 -> 0; the two dipoles' sum against the eikonal 2 C_F s s_12 / (s_13 s_23)
    for x in (1e-3, 1e-5, 1e-7):
        ev = _uug(1 - 0.4 * x, 1 - 0.6 * x)
        ld, on, mem = terms(ev, tok_uug, 0)
        eik = 2 * (4 / 3) * (1 - x) / ((0.6 * x) * (0.4 * x))
        r = (dip(ld, on, mem, 2, 4, 3) + dip(ld, on, mem, 3, 4, 2)) / eik - 1
        assert abs(r) < 3 * x + 1e-6, (x, r)
    # t t~ at one loop: 1/beta and log^2(s/m^2) per top; at tree nothing
    b = 1e-3
    ev = _tt(b)
    ld, on, mem = terms(ev, tok_tt, 1)
    assert int(on.sum()) == 3
    s, m2 = 4 * ev[0, 0] ** 2, 0.1725 ** 2
    got = sorted(float(v) for v in ld[0, on[0]].exp())
    want = sorted([1 / b, np.log(s / m2) ** 2, np.log(s / m2) ** 2])
    assert np.allclose(got, want, rtol=1e-6), (got, want)
    assert int(terms(ev, tok_tt, 0)[1].sum()) == 0
    # model.input_asym_sudakov_single: one more term per top, exp(kappa) log(s/m^2), on that top's coefficient; none at tree
    kap = torch.tensor([np.log(3.0)])
    p1, tp = torch.tensor(ev), torch.tensor([0, len(ev)])
    ld1, on1, mem1 = _asym_terms(p1, tp, len(ev), Gt[tok_tt, 7], Gt[tok_tt, 2] >= 0, torch.tensor([1.0]), kap)
    got1 = sorted(float(v) for v in ld1[0, on1[0]].exp())
    assert np.allclose(got1, sorted(want + [3 * np.log(s / m2)] * 2), rtol=1e-6), got1
    assert sorted(mem1[on1[0]][-2:, 0].tolist()) == [2, 3] and int(_asym_terms(
        p1, tp, len(ev), Gt[tok_tt, 7], Gt[tok_tt, 2] >= 0, torch.tensor([0.0]), kap)[1].sum()) == 0
    # a colourless heavy pair (Z Z) at one QCD loop: no Coulomb, no Sudakov
    assert int(terms(_tt(b, m=91.19), torch.tensor(global_encode(np.array([-11, 11, 23, 23]))), 1)[1].sum()) == 0

    # (2) exactness at fixed bounded readouts: alpha per leg, mu_raw, sd as a trained run might have them
    def gap(ev, tok, loops, lead):
        n = len(ev)
        ld, on, mem = terms(ev, tok, loops)
        alpha = torch.linspace(-2.0, 3.0, n)
        mu = _asym_mu(torch.tensor([4.0]), alpha, torch.tensor([0, n]), n, ld, on, mem, torch.tensor(2.3))
        t = [k for k in range(mem.shape[0]) if tuple(mem[k].tolist()) == lead][0]
        import wrappers
        al = wrappers._ASYM_BOUND * torch.tanh(alpha / wrappers._ASYM_BOUND)
        c = 2.3 * float(al[[q for q in lead if q >= 0]].mean())
        return float(mu) * 2.3 - (c + float(ld[0, t]))
    gaps = [gap(_uug(0.7 + y * 0.3, 1 - y), tok_uug, 0, (2, 4, 3)) for y in (1e-3, 1e-5, 1e-7)]
    assert abs(gaps[-1]) < 1e-5 and abs(gaps[0]) > abs(gaps[-1]), gaps
    gaps = [gap(_tt(b), tok_tt, 1, (2, 3, -1)) for b in (1e-2, 1e-4, 1e-6)]
    assert abs(gaps[-1]) < 1e-4 and abs(gaps[0]) > abs(gaps[-1]), gaps

    # (3) through the wrapper
    def wrap(loss="MSE", gmax=None, **fl):
        from models.lloca import LLOCAMuPTransformer
        ns = 16 + 15 * bool(fl.get("input_pair_logs"))
        torch.manual_seed(0)
        net = LLOCAMuPTransformer(num_scalars=ns, hidden_channels_mlp=16, num_layers_mlp=2, in_channels=ns + 4,
                                  attn_reps="8x0n+2x1n", out_channels=1, num_blocks=2, num_heads=2, loss=loss,
                                  extra_main=int(bool(fl.get("input_asym_factor"))), frames_gamma_max=gmax)
        w = AmplitudeLLoCaWrapper(net, token_size=0, d_particle_hidden=16, **fl)
        w.setup_particle_features(use_pids=False, property_matrix=G)
        if w.input_asym_factor:
            w.asym_sd.fill_(2.3)
        return w.double().eval()

    p = torch.tensor(np.concatenate([_uug(0.9, 0.8), _uug(0.7, 0.95)]))
    ptr = torch.tensor([0, 5, 10])
    tok = torch.cat([tok_uug, tok_uug])
    L = _lorentz(3)
    out = lambda w, x: w(x, tok, 0.0, 1.0, ptr).detach()
    for loss in ("MSE", "HETEROSC"):
        w = wrap(loss, input_pair_logs=6, input_asym_factor=True)
        a, b_ = out(w, p), out(w, p @ L.T)
        assert a.shape == (2, 1 if loss == "MSE" else 2), a.shape
        assert float((a - b_).abs().max() / a.abs().max()) < 1e-7, f"{loss}: not Lorentz invariant"
        assert float((a[:, 0] - out(wrap(loss, input_pair_logs=6), p)[:, 0]).abs().max()) > 1e-3, "the flag does not move mu"
        if loss == "HETEROSC":
            assert w.net.n_main == 2 and w.net.net.linear_out.out_features == 3 and bool((a[:, 1] > 0).all())
    # the single-log flag: a learned scalar, no effect on an event without a heavy coloured leg
    w1 = wrap(input_pair_logs=6, input_asym_factor=True, input_asym_sudakov_single=True)
    assert w1.asym_kappa is not None and w1.asym_kappa.shape == (1,)
    assert torch.allclose(out(w1, p), out(wrap(input_pair_logs=6, input_asym_factor=True), p))
    # the coefficients converge only on a net continuous at the limit: plain momenta (no log inputs) and the frames'
    # boosts capped (model.net.frames_gamma_max=4; uncapped, the relative boosts of a soft or collinear leg grow and the
    # readouts with them, ~y^-0.25 and ~x^-0.5 at this init); then mu's slope is the dipoles' -1 in log y, -2 in log x
    w = wrap(gmax=4.0, input_asym_factor=True)
    one = torch.tensor([0, 5])
    mu = lambda ev: float(w(torch.tensor(ev), tok_uug, 0.0, 1.0, one).detach()) * 2.3
    slope = (mu(_uug(0.7 + 1e-9 * 0.3, 1 - 1e-9)) - mu(_uug(0.7 + 1e-8 * 0.3, 1 - 1e-8))) / np.log(0.1)
    soft = (mu(_uug(1 - 0.4e-9, 1 - 0.6e-9)) - mu(_uug(1 - 0.4e-8, 1 - 0.6e-8))) / np.log(0.1)
    assert abs(slope + 1) < 1e-2 and abs(soft + 2) < 1e-2, (slope, soft)
    print(f"asym_factor: limits, exactness (gap {gaps[-1]:.1e}), wrapper invariance, rows, slopes {slope:.4f} {soft:.4f}: ok")


if __name__ == "__main__":
    for t in [test_asym_factor, test_div_form, test_log_skip, test_route_off, test_scalars_only_reads_no_momentum, test_pair_log_values, test_soft_rescale_fixed_point, test_pair_logs_equivariant,
              test_soft_rescale_equivariant, test_local_logs_equivariant, test_both_flags_6particle_equivariant]:
        t()
    print("all ok")
