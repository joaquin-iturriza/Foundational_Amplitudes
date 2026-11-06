"""CPU checks of the quota-filled flat part (mg5_pipeline_final: flat_quota_bins, flat_quota_weights, flat_quota_select,
_flat_quota_part via build_mixture_dataset with pol["flat_range"]; plan catalog_coverage step 58): the quotas sum to the
flat part; selection keeps exactly the quota per filled bin and spreads an unfilled bin's shortfall over bins with
spare candidates; an empty bin's deficit steers the candidates of the nearest occupied bin, and nothing is steered once
every bin is full; on a toy 2 -> 3 label peaked in the collinear corner the rounds fill every bin up to a top bin the
first pass misses, and the kept events pass the cuts.
    .venv-cpu/bin/python -m pytest tests/test_flat_quota.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mg5_pipeline_final as mg  # noqa: E402


def test_bins_and_quota():
    e, q = mg.flat_quota_bins(-10.0, 0.0, 1003)
    assert q.sum() == 1003 and q.max() - q.min() <= 1
    assert len(q) == int(np.ceil(10.0 / (0.25 * np.log(10.0))))
    e, q = mg.flat_quota_bins(-10.0, 0.0, 50)          # small chunk: at least FLAT_MIN_PER_BIN per bin
    assert len(q) == 5 and q.sum() == 50


def test_select_quota_and_waterfill():
    rng = np.random.default_rng(0)
    edges = np.linspace(0.0, 4.0, 5); quota = np.array([10, 10, 10, 10])
    logm = np.concatenate([rng.uniform(0, 1, 50), rng.uniform(1, 2, 50), rng.uniform(2, 3, 50), rng.uniform(3, 4, 4),
                           [np.nan, 9.0]])
    keep, take = mg.flat_quota_select(logm, edges, quota, rng)
    assert len(keep) == 40 and len(np.unique(keep)) == 40
    assert take[3] == 4 and take[:3].sum() == 36 and take[:3].max() - take[:3].min() <= 1
    assert np.all((logm[keep] >= 0) & (logm[keep] < 4))
    full = np.concatenate([rng.uniform(j, j + 1, 30) for j in range(4)])
    keep, take = mg.flat_quota_select(full, edges, quota, rng)
    assert list(take) == [10, 10, 10, 10]


def test_weights_steer_to_empty_top():
    edges = np.linspace(0.0, 4.0, 5); quota = np.array([5, 5, 5, 5])
    logm = np.array([0.5] * 10 + [1.5] * 10 + [2.5] * 2)      # bin 2 short by 3, bin 3 empty (5 missing)
    w, cnt, deficit = mg.flat_quota_weights(logm, edges, quota)
    assert list(deficit) == [0, 0, 3, 5]
    assert np.all(w[:20] == 0) and np.allclose(w[20:], (3 + 5) / 2)
    logm2 = np.array([0.5] * 10 + [1.5] * 10 + [5.0] * 2)     # beyond the range: steers toward the top edge bin
    w2, _, d2 = mg.flat_quota_weights(logm2, edges, quota)
    # bin 2's deficit is shared by bins 1 and 3 (equally near), bin 3's goes to the two clipped candidates
    assert list(d2) == [0, 0, 5, 5] and np.allclose(w2[20:], 7.5 / 2) and np.allclose(w2[10:20], 2.5 / 10)
    assert np.all(w2[:10] == 0)
    full = np.repeat([0.5, 1.5, 2.5, 3.5], 5)
    assert not mg.flat_quota_weights(full, edges, quota)[0].any()


def _toy_label(ev, sq):
    """|M|^2 ~ 1 / (s * y_min), y_min the least final-pair invariant over s: large in the collinear corner"""
    P = np.stack([e[0] for e in ev])[:, 2:]
    s = sq ** 2
    ys = []
    for i in range(P.shape[1]):
        for j in range(i + 1, P.shape[1]):
            q = P[:, i] + P[:, j]
            ys.append((q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1)) / s)
    return 1.0 / (s * np.clip(np.min(ys, 0), 1e-12, None))


import pytest  # noqa: E402


@pytest.mark.parametrize("multichannel", [False, True])
def test_build_fills_every_bin(multichannel):
    rng = np.random.default_rng(1)
    pdg = [2, -2, -11, 11, 21]; mf = [0.0, 0.0, 0.0]
    cuts = {"pt_min": 10.0, "cos_max": float(np.tanh(5.0)), "dr_min": 0.4, "m_min": -1.0}
    pol = dict(mg.DEFAULT_SAMPLING); pol.update(mode="mixture", f_flat=0.5, oversample=6)
    # the range: the 1% quantile to the maximum of 60000 candidates (reachable by construction), so the top bins need
    # rounds beyond the first one's 1500
    sampler = mg._sqrts_sampler(30.0, 6000.0, "log", rng)
    ev, sq = mg._candidates(60000, sampler, mf, pdg, rng, cuts, 0.5, pol["y_lo"])
    lm = np.log(_toy_label(ev, sq))
    pol["flat_range"] = [float(np.quantile(lm, 0.01)), float(lm.max())]
    if multichannel:
        pol.update(multichannel=True, mc_resonances=[[[0, 1], 23]], mc_isc=True, mc_res_params={23: mg.MC_RES[23]})
    events, sqrts, me2 = mg.build_mixture_dataset(1000, 30.0, 6000.0, mf, pdg, rng, cuts, pol, _toy_label,
                                                  sqrts_measure="log")
    assert len(events) == 1000
    d = mg.FLAT_LAST
    assert d["kept_per_bin"].sum() == 500 and np.all(d["kept_per_bin"] >= d["quota"])
    assert len(d["rounds"]) >= 2 and d["rounds"][0]["bins_short"] > 0 and d["rounds"][-1]["missing"] == 0
    assert d["n_candidates"] <= mg.FLAT_BUDGET * 500
    P = np.stack([e[0] for e in events])
    assert mg.fiducial_pass_mask(P, mf, pdg, cuts).all()
    if multichannel:
        assert d["channels"] and len(np.unique(d["chan"])) == len(d["channels"])
