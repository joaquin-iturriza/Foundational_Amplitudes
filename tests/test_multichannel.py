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


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
