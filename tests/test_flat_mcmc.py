"""The Markov-chain flat part (mg5_pipeline_final._flat_mcmc_part, sampling.flat_mcmc): the pair move keeps every
mass, the pair's mass, the other finals and the total momentum; on a toy |M|^2 with a collinear and a soft-like pole
the chains, at the same label budget as the quota sampler, fill more of the quota without near-copies of one state
(analysis/catalog_v4/flat_diversity.py: no more than 2% of kept events have another within 1% in every invariant)."""
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mg5_pipeline_final as mg  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "catalog_v4"))
import flat_diversity as fd  # noqa: E402

M = [0.0, 0.0, 0.0, 0.0]
PDG = [2, -2, 21, 21, 1, -1]


def _mass2(p):
    return p[..., 0] ** 2 - (p[..., 1:] ** 2).sum(-1)


def toy_label(ev, sq):
    P = np.stack([e[0] for e in ev])
    s = lambda a, b: np.abs(_mass2(P[:, a] + P[:, b]))
    t = lambda a, b: np.abs(_mass2(P[:, a] - P[:, b]))
    return 1.0 / (s(2, 3) * s(4, 5) * t(0, 2) + 1e-30) * 1e8


def _pol(budget, mcmc):
    return dict(flat_range=[np.log(1e-8), np.log(1e9)], flat_dex_per_bin=0.25, flat_budget=budget, oversample=6,
                y_lo=1e-6, democratic=0.5, multichannel=False, flat_mcmc=mcmc, n_sqrts_bins=200)


def test_pair_move_invariants():
    rng = np.random.default_rng(1)
    ev, sq = mg._candidates(200, lambda k: np.full(k, 500.0), M, PDG, rng, None, 0.5, 1e-6)
    P = np.stack([e[0] for e in ev])
    Pn = mg._pair_rotate(P, 2, 4, rng)
    assert np.allclose(Pn[:, 2:].sum(1), P[:, 2:].sum(1), atol=1e-8 * 500)
    assert np.allclose(_mass2(Pn[:, 2:]), 0.0, atol=1e-6 * 500 ** 2)
    assert np.allclose(_mass2(Pn[:, 2] + Pn[:, 4]), _mass2(P[:, 2] + P[:, 4]), rtol=1e-8, atol=1e-6)
    assert np.allclose(Pn[:, [0, 1, 3, 5]], P[:, [0, 1, 3, 5]])
    assert not np.allclose(Pn[:, 2], P[:, 2])


def fill(mcmc, budget, seed=3, n_flat=2000):
    rng = np.random.default_rng(seed)
    uni = lambda k: np.exp(rng.uniform(np.log(50.0), np.log(3000.0), size=k))
    ev, sq, me = (mg._flat_mcmc_part(n_flat, uni, M, PDG, rng, None, _pol(budget, True), toy_label, 0.0, None) if mcmc
                  else mg._flat_quota_part(n_flat, 50.0, 3000.0, uni, True, M, PDG, rng, None, _pol(budget, False),
                                           toy_label, 0.0, None))
    d = mg.FLAT_LAST
    F = fd.features(d["kept_P"], d["kept_sq"])
    pb = fd.per_bin(F, fd.bins_of(d["logm"][d["keep"]], d["edges"]), d["kept_chain"], len(d["quota"]))
    dup = sum(x.get("dup", 0.0) * x["n"] for x in pb) / max(sum(x["n"] for x in pb), 1)
    return float(np.minimum(d["kept_per_bin"], d["quota"]).sum() / d["quota"].sum()), int(d["n_labels"]), dup


def test_mcmc_fills_toy():
    f_q, n_q, dup_q = fill(False, 12)
    f_m, n_m, dup_m = fill(True, 12)
    print("quota budget 12", f_q, n_q, dup_q, "| mcmc budget 12", f_m, n_m, dup_m)
    assert n_m <= 12 * 2000
    assert dup_m <= 0.02
    assert f_m > f_q


if __name__ == "__main__":
    test_pair_move_invariants(); test_mcmc_fills_toy(); print("ok")
