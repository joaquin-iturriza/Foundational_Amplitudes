"""CPU checks of the quota-filled flat part (mg5_pipeline_final: flat_quota_bins, flat_quota_weights, flat_quota_select,
_flat_quota_part via build_mixture_dataset with pol["flat_range"]; plan catalog_coverage step 58): the quotas sum to the
flat part and their remainder favours no bin over many chunks; the bin width and the budget enter the recipe id only
with flat_range; a chunk stopped at the budget with bins short says so; selection keeps exactly the quota per filled bin and spreads an unfilled bin's shortfall over bins with
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
    rng = np.random.default_rng(0)
    e, q = mg.flat_quota_bins(-10.0, 0.0, 1003, rng)
    assert q.sum() == 1003 and q.max() - q.min() <= 1
    assert len(q) == int(np.ceil(10.0 / (0.25 * np.log(10.0)))) and np.isclose(e[0], -10.0) and np.isclose(e[-1], 0.0)
    e, q = mg.flat_quota_bins(-10.0, 0.0, 10, rng)      # fewer events than bins: zeros in this chunk only
    assert q.sum() == 10 and set(np.unique(q)) <= {0, 1}
    e, q = mg.flat_quota_bins(-10.0, 0.0, 1003, rng, dex_per_bin=0.5)
    assert len(q) == int(np.ceil(10.0 / (0.5 * np.log(10.0))))


def test_remainder_not_biased():
    """over many chunks the remainder lands on every bin alike: the top bins are not systematically short"""
    rng = np.random.default_rng(5)
    tot = sum(mg.flat_quota_bins(-10.0, 0.0, 1003, rng)[1] for _ in range(4000))
    nb = len(tot); per = 1003 // nb; rem = 1003 - per * nb
    extra = tot - 4000 * per                       # times each bin got the remainder's +1
    exp = 4000 * rem / nb; sd = np.sqrt(4000 * (rem / nb) * (1 - rem / nb))
    assert np.all(np.abs(extra - exp) < 5 * sd)
    assert abs(extra[-nb // 4:].mean() - extra[:nb // 4].mean()) < 3 * sd


def test_constants_in_recipe_id(monkeypatch):
    """with flat_range the bin width and the budget are in the policy, so in the recipe id; without it neither is"""
    base = next(n for n, c in mg.PROCESSES.items() if c.get("kind") not in ("virt", "lhe_pp") and c.get("pdg_ids")
                and not mg.is_synthetic(n))
    def rid(sampling):
        monkeypatch.setitem(mg.PROCESSES, "zz_flat_quota_test", dict(mg.PROCESSES[base], sampling=sampling))
        return mg.recipe_id(mg.variable_energy_recipe("zz_flat_quota_test", 100.0, 6000.0, 1000, role="train", seed=42))
    s0 = {"mode": "mixture"}
    assert "flat_budget" not in mg.sampling_policy({"sampling": s0}) and \
        "flat_dex_per_bin" not in mg.sampling_policy({"sampling": s0})
    s1 = dict(s0, flat_range=[-10.0, 0.0])
    pol = mg.sampling_policy({"sampling": s1})
    assert pol["flat_budget"] == mg.FLAT_BUDGET and pol["flat_dex_per_bin"] == mg.FLAT_DEX_PER_BIN
    ids = {rid(s0), rid(s1), rid(dict(s1, flat_range=[-10.0, 1.0])), rid(dict(s1, flat_budget=40)),
           rid(dict(s1, flat_dex_per_bin=0.5))}
    assert len(ids) == 5
    assert rid(s1) == rid(dict(s1, flat_budget=mg.FLAT_BUDGET, flat_dex_per_bin=mg.FLAT_DEX_PER_BIN))
    monkeypatch.setattr(mg, "FLAT_BUDGET", 40)     # a changed module default changes the id too
    assert rid(s1) == rid(dict(s1, flat_budget=40))


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
    samp = dict(mode="mixture", f_flat=0.5, oversample=6)
    pol = dict(mg.DEFAULT_SAMPLING); pol.update(samp)
    # the range: the 1% quantile to the maximum of 60000 candidates (reachable by construction), so the top bins need
    # rounds beyond the first one's 1500
    sampler = mg._sqrts_sampler(30.0, 6000.0, "log", rng)
    ev, sq = mg._candidates(60000, sampler, mf, pdg, rng, cuts, 0.5, pol["y_lo"])
    lm = np.log(_toy_label(ev, sq))
    samp["flat_range"] = [float(np.quantile(lm, 0.01)), float(lm.max())]
    if multichannel:
        samp.update(multichannel=True, mc_resonances=[[[0, 1], 23]], mc_isc=True, mc_res_params={23: mg.MC_RES[23]})
    pol = mg.sampling_policy({"sampling": samp})
    events, sqrts, me2 = mg.build_mixture_dataset(1000, 30.0, 6000.0, mf, pdg, rng, cuts, pol, _toy_label,
                                                  sqrts_measure="log")
    assert len(events) == 1000
    d = mg.FLAT_LAST
    assert d["kept_per_bin"].sum() == 500 and np.all(d["kept_per_bin"] >= d["quota"]) and not d["budget_hit"]
    assert len(d["rounds"]) >= 2 and d["rounds"][0]["bins_short"] > 0 and d["rounds"][-1]["missing"] == 0
    assert d["n_candidates"] <= mg.FLAT_BUDGET * 500
    P = np.stack([e[0] for e in events])
    assert mg.fiducial_pass_mask(P, mf, pdg, cuts).all()
    if multichannel:
        assert d["channels"] and len(np.unique(d["chan"])) == len(d["channels"])


def test_budget_hit_reported(capsys):
    """a range whose top no candidate reaches: the chunk stops at the budget and reports it"""
    rng = np.random.default_rng(2)
    pdg = [2, -2, -11, 11, 21]; mf = [0.0, 0.0, 0.0]
    cuts = {"pt_min": 10.0, "cos_max": float(np.tanh(5.0)), "dr_min": 0.4, "m_min": -1.0}
    pol = mg.sampling_policy({"sampling": dict(mode="mixture", f_flat=0.5, oversample=6, flat_range=[-30.0, 60.0],
                                               flat_budget=8)})
    mg.build_mixture_dataset(400, 30.0, 6000.0, mf, pdg, rng, cuts, pol, _toy_label, sqrts_measure="log")
    d = mg.FLAT_LAST
    assert d["budget_hit"] and d["n_candidates"] == 8 * 200 and d["kept_per_bin"].sum() == 200
    assert "[WARN] flat quota: stopped at the budget" in capsys.readouterr().out
