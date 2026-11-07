"""CPU checks of the multichannel flat-part candidates (mg5_pipeline_final: resonance_draw, is_collinear_draw,
_candidates_mc, mc_shares_from_pass, resonance_channels; plan catalog_coverage step 57): every channel conserves
momentum and keeps every final on its mass shell; a resonance channel puts most of its events' pair mass within a few
widths of the pole; the collinear channel reaches |eta| between 4 and 5 in the partonic frame; the second-pass
shares favour the channel that holds the rare log|M|^2 values and keep the floor; resonance_channels finds an
s-channel Z decaying to two of three finals in a hand-made diagram; the joint-resonance and multiperipheral
channels (step 58's re-check) are exact in phase space and reach their regions.
    .venv-cpu/bin/python -m pytest tests/test_multichannel.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mg5_pipeline_final as mg  # noqa: E402


# trunk's resonance_draw (inlined Breit-Wigner, @fcb06e7a) at default_rng(11), 5 events at 500 GeV: m(0, 1)
PINNED_MLL = [88.46533766507697, 91.20103009081356, 91.60378136887829, 80.03982510431064, 36.696173689799515]


def _mass(p):
    return np.sqrt(np.clip(p[..., 0] ** 2 - (p[..., 1:] ** 2).sum(-1), 0, None))


def _check_ps(P, sq, masses):
    tot = P.sum(1)
    assert np.allclose(tot[:, 0], sq, rtol=1e-9, atol=1e-7)
    assert np.allclose(tot[:, 1:], 0, atol=1e-7 * sq.max())
    m2 = P[..., 0] ** 2 - (P[..., 1:] ** 2).sum(-1)
    assert np.allclose(m2, np.asarray(masses)[None, :] ** 2, atol=1e-7 * sq.max() ** 2)


def test_resonance_draw():
    rng = np.random.default_rng(0)
    sq = np.exp(rng.uniform(np.log(120.0), np.log(3000.0), 4000))
    masses = [0.0, 0.0, 0.0]
    P = mg.resonance_draw(4000, sq, masses, (0, 1), 91.188, 2.441404, 1e-6, rng)
    _check_ps(P, sq, masses)
    mll = _mass(P[:, 0] + P[:, 1])
    assert (np.abs(mll - 91.188) < 5 * 2.441404).mean() > 0.6     # 0.8 on the Breit-Wigner


def test_resonance_draw_massive_and_four_finals():
    rng = np.random.default_rng(1)
    sq = np.exp(rng.uniform(np.log(400.0), np.log(3000.0), 2000))
    masses = [0.0, 80.419, 0.0, 0.0]
    P = mg.resonance_draw(2000, sq, masses, (0, 1), 172.5, 1.4915, 1e-6, rng)
    _check_ps(P, sq, masses)
    mt = _mass(P[:, 0] + P[:, 1])
    assert (np.abs(mt - 172.5) < 10 * 1.4915).mean() > 0.6


def test_is_collinear_draw():
    rng = np.random.default_rng(2)
    sq = np.exp(rng.uniform(np.log(1600.0), np.log(6000.0), 4000))   # where |eta| 5 at pT 10 GeV is reachable
    masses = [0.0, 0.0, 0.0, 0.0]
    P = mg.is_collinear_draw(4000, sq, masses, (2, 3), 1e-6, rng)
    _check_ps(P, sq, masses)
    pt = np.hypot(P[..., 1], P[..., 2])
    eta = np.arcsinh(P[..., 3] / np.clip(pt, 1e-300, None))
    big = np.abs(eta[:, 2:]).max(1)
    # the drawn parton's |eta| is uniform to 5 (the recoil's finals may go beyond; the cuts then remove the event)
    assert ((big > 4) & (big <= 5)).mean() > 0.1


def test_candidates_mc_and_shares():
    rng = np.random.default_rng(3)
    pdg = [2, -2, -11, 11, 21]
    ch = mg.mc_channels([0.0, 0.0, 0.0], pdg, [((0, 1), 23)])
    assert [c[0] for c in ch] == ["rambo", "splitter", "res", "isc"]
    cuts = {"pt_min": 10.0, "cos_max": float(np.tanh(5.0)), "dr_min": 0.4, "m_min": 0.0}
    sampler = mg._sqrts_sampler(30.0, 6000.0, "log", rng)
    ev, sq, c = mg._candidates_mc(3000, sampler, [0.0, 0.0, 0.0], pdg, rng, cuts, ch, [0.25] * 4, 1e-6, 1 / 3)
    assert len(ev) == 3000 and set(np.unique(c)) <= set(range(4)) and len(np.unique(c)) == 4
    P = np.stack([e[0] for e in ev])
    assert mg.fiducial_pass_mask(P, [0.0, 0.0, 0.0], pdg, cuts).all()
    logm = rng.normal(0, 1, 4000); chan = np.repeat(np.arange(4), 1000)
    logm[chan == 2] = rng.uniform(5, 10, 1000)
    sh = mg.mc_shares_from_pass(chan, logm, 4)
    assert sh[2] == sh.max() and sh.min() > 0 and abs(sh.sum() - 1) < 1e-12


def test_resonance_channels():
    # u u~ > e+ e- g: u u~ -> (v5) gluon off a quark line (v6), the quark line -> Z (v6-v7), Z -> e+ e- at v7
    nodes = [{"kind": "external", "leg_number": i} for i in range(1, 6)] + [{"kind": "vertex"}] * 3
    edges = [{"u": 0, "v": 5, "pdg": 2, "external": True}, {"u": 1, "v": 6, "pdg": -2, "external": True},
             {"u": 4, "v": 5, "pdg": 21, "external": True}, {"u": 5, "v": 6, "pdg": 2, "external": False},
             {"u": 6, "v": 7, "pdg": 23, "external": False},
             {"u": 2, "v": 7, "pdg": -11, "external": True}, {"u": 3, "v": 7, "pdg": 11, "external": True}]
    sc = {"subprocesses": [{"external": [{"number": i + 1, "pdg": p} for i, p in enumerate([2, -2, -11, 11, 21])],
                            "diagrams": [{"nodes": nodes, "edges": edges, "diagram_type": "tree"}]}]}
    # the pool stores the finals as [e-, e+, g] (slot order differs from the legs): the Z holds slots 2, 3 still
    assert mg.resonance_channels(sc, [2, -2, 11, -11, 21]) == [((0, 1), 23)]
    assert mg.resonance_channels(sc, [2, -2, -11, 11, 21]) == [((0, 1), 23)]



def test_mc_res_params():
    # the heavy-quark layer's top (recipes/select_catalog_v4.py: mass with its LO width) moves the top channel
    rp = mg.mc_res_params({"masses": {6: 300.0}, "widths": {6: 8.729709}})
    assert rp[6] == (300.0, 8.729709) and rp[23] == mg.MC_RES[23]
    ch = mg.mc_channels([0.0, 0.0, 0.0], [2, -2, -11, 11, 21], [((0, 1), 23)], res_params=mg.mc_res_params(None))
    assert ch[2] == ("res", (0, 1), 23) + mg.MC_RES[23]
    try:
        mg.mc_res_params({"masses": {6: 300.0}})
    except ValueError:
        pass
    else:
        raise AssertionError("a top mass without its width must be refused")


def test_resonance2_draw():
    """two resonances on disjoint finals on shell together: phase space exact, both pair masses near their poles"""
    rng = np.random.default_rng(3)
    sq = np.exp(rng.uniform(np.log(250.0), np.log(3000.0), 4000))
    masses = [0.0, 0.0, 0.0, 0.0]
    P = mg.resonance2_draw(4000, sq, masses, (0, 1), 91.188, 2.441404, (2, 3), 80.419, 2.0476, 1e-6, rng)
    _check_ps(P, sq, masses)
    both = (np.abs(_mass(P[:, 0] + P[:, 1]) - 91.188) < 5 * 2.441404) & \
           (np.abs(_mass(P[:, 2] + P[:, 3]) - 80.419) < 5 * 2.0476)
    assert both.mean() > 0.45      # 0.8 x 0.8 on both Breit-Wigners
    # with a recoil final, and massive
    masses = [0.0, 0.0, 0.0, 0.0, 172.5]
    sq = np.exp(rng.uniform(np.log(500.0), np.log(3000.0), 2000))
    P = mg.resonance2_draw(2000, sq, masses, (0, 1), 91.188, 2.441404, (2, 3), 91.188, 2.441404, 1e-6, rng)
    _check_ps(P, sq, masses)


def test_multiperipheral_draw():
    """every final at pT near the cut, spread in rapidity to the acceptance edge at large sqrt(s); phase space exact"""
    rng = np.random.default_rng(4)
    sq = np.exp(rng.uniform(np.log(50.0), np.log(6000.0), 4000))
    for masses in ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 80.419]):
        P = mg.multiperipheral_draw(4000, sq, masses, 1e-6, rng)
        _check_ps(P, sq, masses)
    masses = [0.0] * 4
    sq = np.full(3000, 1500.0)
    P = mg.multiperipheral_draw(3000, sq, masses, 1e-6, rng)
    _check_ps(P, sq, masses)
    pt = np.hypot(P[..., 1], P[..., 2]); pa = np.linalg.norm(P[..., 1:], axis=-1)
    eta = 0.5 * np.log((pa + P[..., 3]) / (pa - P[..., 3]))
    assert np.median(np.abs(eta).max(1)) > 3.5 and (np.abs(eta).max(1) > 4.5).mean() > 0.05
    assert np.median(pt.min(1)) < 25.0
    # inside the acceptance after the boost, also where the span is capped (5 TeV), and above the splitter's there
    pdg = np.array([2, 3, 1, -1, 3, 2])
    cuts = mg.V4_CUTS["hadronic"]
    for e in (1500.0, 5000.0):
        sq = np.full(3000, e)
        P = mg.multiperipheral_draw(3000, sq, masses, 1e-6, rng)
        _check_ps(P, sq, masses)
        pa = np.linalg.norm(P[..., 1:], axis=-1)
        assert (np.abs(0.5 * np.log((pa + P[..., 3]) / (pa - P[..., 3]))).max(1) <= 5.0 + 1e-9).all()
        full = np.concatenate([mg._beams(sq, pdg, None), P], 1)
        acc = mg.fiducial_pass_mask(full, np.zeros(4), pdg, cuts).mean()
        spl = mg.democratic_draw(3000, sq, masses, 1e-6, rng)
        acc_s = mg.fiducial_pass_mask(np.concatenate([mg._beams(sq, pdg, None), spl], 1), np.zeros(4), pdg, cuts).mean()
        print(f"ladder acceptance at {e:.0f} GeV {acc:.3f}, splitter {acc_s:.3f}")
        assert acc > 0.7 and acc > 2 * acc_s


PINNED_MPERI_PZ = [-5.884409088916563, 2.6553593937263056, 0.7716145554661688, 2.4574351397118717, -0.3205210552266862,
                   -45.19052783313533, 136.19802666363015, -90.68697777526815, 704.8498443536913, -17.114595845210008,
                   5.284393600968798, -693.0196421094499, -1110.8624996625797, -1222.7408353322023, 2497.99741511791,
                   -164.39408012312714]


def test_multiperipheral_edge():
    """plan step 66: without `edge` the step-58 ladder's draws (pinned from trunk 1393e8e9); with it phase space exact,
    inside |eta| < 5 and the acceptance as the old ladder's, and the corner (every pT at the cut, the span at the edge,
    sqrt(s) ~ 2 x 10 GeV x cosh 5) proposed far more often"""
    rng = np.random.default_rng(21)
    P = mg.multiperipheral_draw(4, np.array([60.0, 300.0, 1450.0, 5000.0]), [0.0] * 4, 1e-6, rng)
    np.testing.assert_allclose(P[:, :, 3].ravel(), PINNED_MPERI_PZ, rtol=1e-12)
    rng = np.random.default_rng(5)
    sq = np.exp(rng.uniform(np.log(50.0), np.log(6000.0), 4000))
    for masses in ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 80.419]):
        _check_ps(mg.multiperipheral_draw(4000, sq, masses, 1e-6, rng, edge=True), sq, masses)
    masses, pdg, cuts = [0.0] * 4, np.array([2, 3, 1, -1, 3, 2]), mg.V4_CUTS["hadronic"]
    corner = {}
    for edge in (False, True):
        sq = np.full(20000, 1450.0)
        P = mg.multiperipheral_draw(20000, sq, masses, 1e-6, np.random.default_rng(1), edge=edge)
        _check_ps(P, sq, masses)
        pt = np.hypot(P[..., 1], P[..., 2]); pa = np.linalg.norm(P[..., 1:], axis=-1)
        eta = np.abs(0.5 * np.log((pa + P[..., 3]) / (pa - P[..., 3])))
        assert (eta.max(1) <= 5.0 + 1e-9).all()
        acc = mg.fiducial_pass_mask(np.concatenate([mg._beams(sq, pdg, None), P], 1), np.zeros(4), pdg, cuts).mean()
        corner[edge] = ((pt < 13.0).all(1) & (eta.max(1) > 4.8)).mean()
        print(f"edge={edge}: acceptance {acc:.3f}, every pT < 13 GeV and max|eta| > 4.8: {corner[edge]:.4f}, "
              f"first three pT < 11 GeV: {(pt[:, :3] < 11.0).all(1).mean():.4f}")
        assert acc > 0.7
    assert corner[True] > 0.03 and corner[True] > 20 * corner[False]
    ch = mg.mc_channels([0.0] * 4, [2, 2, 5, -5, 2, 2], [], mperi=True, mperi_edge=True)
    assert ch[-1] == ("mperi", "edge") and not any(c == ("mperi",) for c in ch)
    both = mg.mc_channels([0.0] * 4, [2, 2, 5, -5, 2, 2], [], mperi=True, mperi_both=True)
    assert both[-2:] == [("mperi",), ("mperi", "edge")]
    sq = np.full(500, 1450.0)
    _check_ps(mg._mc_draw(ch[-1], 500, sq, np.zeros(4), 1e-6, np.random.default_rng(2)), sq, masses)


def test_mc_channels_flags():
    """res2 pairs only disjoint resonances; mperi needs two partons; both off by default"""
    res = [[(0, 1), 23], [(2, 3), 24], [(1, 2), 24]]
    pdg = [2, -2, 11, -11, 12, -12]
    base = mg.mc_channels([0.0] * 4, pdg, res)
    assert not any(c[0] in ("res2", "mperi") for c in base)
    ch = mg.mc_channels([0.0] * 4, pdg, res, res2=True, mperi=True)
    r2 = [c for c in ch if c[0] == "res2"]
    assert len(r2) == 1 and r2[0][1] == (0, 1) and r2[0][5] == (2, 3)
    assert not any(c[0] == "mperi" for c in ch)          # no parton among the finals
    ch = mg.mc_channels([0.0] * 4, [2, 2, 5, -5, 2, 2], [], mperi=True)
    assert ch[-1] == ("mperi",)
    assert not any(c[0] == "mperi" for c in mg.mc_channels([0.0] * 4, [11, -11, 1, -1, 2, -2], [], mperi=True))


def test_resonance_draw_unchanged():
    """resonance_draw through _bw_mass draws what the inlined Breit-Wigner drew (same order of draws): pinned values"""
    rng = np.random.default_rng(11)
    P = mg.resonance_draw(5, np.full(5, 500.0), [0.0] * 3, (0, 1), 91.188, 2.441404, 1e-6, rng)
    np.testing.assert_allclose(_mass(P[:, 0] + P[:, 1]), PINNED_MLL, rtol=1e-12)


def test_rotate_forward_free_floor():
    """sampling.fwd_free_floor (step (f)): the forward component's picked final reaches 1 - |cos theta| down to the floor
    when no cut applies to it (a neutrino, a massive final), stays at the cut edge when it is a cut leg, keeps momentum
    and mass shells; without the flag the draws are the earlier ones (same random stream)"""
    m = [0.0, 0.0, 91.188]
    pdg = [11, -11, 14, -14, 23]
    cuts = dict(mg.V4_CUTS["ee"])
    assert mg.fwd_free_mask(m, pdg, cuts) is None
    free = mg.fwd_free_mask(m, pdg, dict(cuts, fwd_free_floor=1e-8))
    assert free.tolist() == [True, True, True]
    assert mg.fwd_free_mask([0.0, 0.0, 0.0], [11, -11, 15, -15, 22], dict(cuts, fwd_free_floor=1e-8)).tolist() == \
        [False, False, False]
    n, sq = 4000, np.full(4000, 3000.0)
    F = mg.democratic_draw(n, sq, m, 1e-6, np.random.default_rng(3))
    P = np.concatenate([mg._beams(sq, np.asarray(pdg), None), F], 1)
    a = mg.rotate_forward(P, 0.99, np.random.default_rng(5))
    b = mg.rotate_forward(P, 0.99, np.random.default_rng(5), None, None)
    np.testing.assert_array_equal(a, b)
    c = mg.rotate_forward(P, 0.99, np.random.default_rng(5), free, 1e-8)
    _check_ps(c[:, 2:], sq, m)
    cth = np.abs(c[:, 2:, 3]) / np.linalg.norm(c[:, 2:, 1:], axis=-1)
    d = np.log10(1.0 - cth.max(1))
    assert (d < -2).mean() > 0.5 and d.min() < -7.0          # 8 decades log-uniform: most below the 0.99 cut edge
    cut_all = np.zeros(3, bool)
    e = mg.rotate_forward(P, 0.99, np.random.default_rng(5), cut_all, 1e-8)
    np.testing.assert_array_equal(e, a)                       # every final cut: the earlier draws exactly


BEFORE_SOLVE = "b3ba8b94"   # trunk before the ladder's solve (plan step 72's cost fix): its 60 halvings are the reference


def _module_at(rev, name):
    import importlib.util, subprocess, tempfile
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = subprocess.run(["git", "-C", here, "show", f"{rev}:mg5_pipeline_final.py"], capture_output=True, text=True)
    if src.returncode:
        import pytest
        pytest.skip(f"git show {rev} failed: {src.stderr.strip()}")
    path = os.path.join(tempfile.mkdtemp(prefix="zz_probe_solve_"), f"{name}.py")
    open(path, "w").write(src.stdout)
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def test_multiperipheral_solve_equals_halvings():
    """the ladder's regula-falsi solve returns the 60 halvings' momenta to rounding (same random numbers), on massless
    and massive finals, 3 to 5 of them, both variants, sqrt(s) over the whole window"""
    old = _module_at(BEFORE_SOLVE, "mg_before_solve")
    for masses in ([0.0] * 4, [0.0] * 3, [0.0] * 5, [91.188, 0.0, 0.0], [80.419, 0.0, 0.0, 0.0]):
        for edge in (False, True):
            sq = np.exp(np.random.default_rng(7).uniform(np.log(40.0), np.log(6000.0), 3000))
            a = old.multiperipheral_draw(3000, sq, masses, 1e-6, np.random.default_rng(5), edge=edge)
            b = mg.multiperipheral_draw(3000, sq, masses, 1e-6, np.random.default_rng(5), edge=edge)
            scale = np.abs(a).max(axis=(1, 2), keepdims=True)
            assert np.max(np.abs(a - b) / scale) < 1e-12, (masses, edge)


def test_resonance2_forward():
    """res2f: phase space exact, both poles as the isotropic channel, the pair's direction log-forward (half of the
    draws at 1 - |cos theta| < 1e-3 against 1e-3 isotropic, down to the floor); without fwd the draws are unchanged"""
    rng = np.random.default_rng(3)
    sq = np.exp(rng.uniform(np.log(250.0), np.log(3000.0), 4000))
    masses = [0.0] * 4
    P = mg.resonance2_draw(4000, sq, masses, (0, 1), 91.188, 2.441404, (2, 3), 80.419, 2.0476, 1e-6, rng, fwd=True)
    _check_ps(P, sq, masses)
    Z = P[:, 0] + P[:, 1]
    both = (np.abs(_mass(Z) - 91.188) < 5 * 2.441404) & (np.abs(_mass(P[:, 2] + P[:, 3]) - 80.419) < 5 * 2.0476)
    assert both.mean() > 0.45
    one_m = 1.0 - np.abs(Z[:, 3]) / np.linalg.norm(Z[:, 1:], axis=1)
    ok = both
    assert 0.4 < (one_m[ok] < 1e-3).mean() < 0.6 and one_m[ok].min() < 1e-5
    assert 0.4 < (Z[ok, 3] > 0).mean() < 0.6
    a = mg.resonance2_draw(50, sq[:50], masses, (0, 1), 91.188, 2.441404, (2, 3), 80.419, 2.0476, 1e-6,
                           np.random.default_rng(9))
    b = mg.resonance2_draw(50, sq[:50], masses, (0, 1), 91.188, 2.441404, (2, 3), 80.419, 2.0476, 1e-6,
                           np.random.default_rng(9), fwd=False)
    assert np.array_equal(a, b)


def test_mc_channels_res2_fwd():
    """res2_fwd adds one res2f per joint pair, after the res2 ones; nothing without res2 or without a disjoint pair"""
    res = [[(0, 1), 23], [(2, 3), 24], [(1, 2), 24]]
    pdg = [2, -2, 11, -11, 12, -12]
    ch = mg.mc_channels([0.0] * 4, pdg, res, res2=True, res2_fwd=True)
    r2 = [c for c in ch if c[0] == "res2"]; r2f = [c for c in ch if c[0] == "res2f"]
    assert len(r2f) == 1 and r2f[0][1:] == r2[0][1:]
    assert ch == mg.mc_channels([0.0] * 4, pdg, res, res2=True) + r2f
    assert not any(c[0] == "res2f" for c in mg.mc_channels([0.0] * 4, pdg, res, res2_fwd=True))


def test_sampling_policy_mc_constants():
    """multichannel policies carry the proposal's constants (so the recipe id follows them); a policy naming another
    value than the code's is refused; a policy without multichannel is unchanged"""
    base = dict(mode="mixture", f_flat=0.5, oversample=6)
    assert not any(k.startswith("mc_") for k in mg.sampling_policy({"sampling": base}))
    pol = mg.sampling_policy({"sampling": dict(base, multichannel=True, mc_mperi_both=True, mc_res2_fwd=True)})
    assert pol["mc_tail"] == mg.MC_TAIL and pol["mc_res2_fwd_floor"] == mg.MC_RES2_FWD_FLOOR
    assert pol["mc_ladder_pt_span"] == mg.MC_LADDER_PT_SPAN and pol["mc_ladder_edge_d"] == list(mg.MC_LADDER_EDGE_D)
    assert pol["mc_res_cards"]["23"] == list(mg.MC_RES[23])
    plain = mg.sampling_policy({"sampling": dict(base, multichannel=True)})
    assert "mc_res2_fwd_floor" not in plain and "mc_ladder_pt_span" not in plain and "mc_tail" in plain
    assert mg.sampling_policy({"sampling": dict(base, multichannel=True, mc_tail=mg.MC_TAIL)})["mc_tail"] == mg.MC_TAIL
    try:
        mg.sampling_policy({"sampling": dict(base, multichannel=True, mc_res2_fwd=True, mc_res2_fwd_floor=1e-4)})
        raise AssertionError("a floor other than the code's must be refused")
    except ValueError:
        pass
    # round trips of the policy (the recipe sidecar sorts keys, YAML too) and int-for-float values are accepted
    import json, yaml
    assert mg.sampling_policy({"sampling": json.loads(json.dumps(pol))}) == pol
    assert mg.sampling_policy({"sampling": json.loads(json.dumps(pol, sort_keys=True))}) == pol
    assert mg.sampling_policy({"sampling": yaml.safe_load(yaml.safe_dump(pol))}) == pol
    assert mg.sampling_policy({"sampling": dict(base, multichannel=True, mc_pt_lo=10)})["mc_pt_lo"] == mg.MC_PT_LO
    g = mg.sampling_policy({"sampling": dict(base, flat_range=[-5.0, 5.0], flat_grid=True)})
    assert g["flat_grid_smooth"] == mg.FLAT_GRID_SMOOTH
    try:
        mg.sampling_policy({"sampling": dict(base, flat_range=[-5.0, 5.0], flat_grid=True, flat_grid_smooth=5)})
        raise AssertionError("a smoothing other than the code's must be refused")
    except ValueError:
        pass


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
