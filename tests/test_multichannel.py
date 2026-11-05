"""CPU checks of the multichannel flat-part candidates (mg5_pipeline_final: resonance_draw, is_collinear_draw,
_candidates_mc, mc_shares_from_pass, resonance_channels; plan catalog_coverage step 57): every channel conserves
momentum and keeps every final on its mass shell; a resonance channel puts most of its events' pair mass within a few
widths of the pole; the collinear channel reaches |eta| between 4 and 5 in the partonic frame; the second-pass
shares favour the channel that holds the rare log|M|^2 values and keep the floor; resonance_channels finds an
s-channel Z decaying to two of three finals in a hand-made diagram.
    .venv-cpu/bin/python -m pytest tests/test_multichannel.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mg5_pipeline_final as mg  # noqa: E402


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


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
