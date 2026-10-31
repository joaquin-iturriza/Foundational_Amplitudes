"""lhe.label_alphas_mz: an LHE pool labelled at alpha_s(sqrt(s_hat)) run from that alpha_s(M_Z) (the H6 held-out
class's 0.13, recipes/gen_catalog_v4_heldout_default.py); absent, the 0.118 running as before."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mg5_pipeline_final as mg  # noqa: E402

NAME = "zz_label_amz"


def _finish(lhe_extra):
    mg.PROCESSES[NAME] = {"kind": "lhe_pp", "pdg_ids": [21] * 5, "m_finals": [0.0] * 3, "nfinal": 3,
                          "mg5_generate": ["generate g g > g g g"],
                          "lhe": {"cards_dir": "recipes/catalog_v4_pp", "run_card": "run_card.dat",
                                  "param_card": "param_card.dat", "label_alphas": "running_sqrt_shat",
                                  "n_val": 0, "n_test": 50, "n_train": 0, **lhe_extra}}
    rng = np.random.default_rng(0)
    n = 50
    m = np.zeros((n, 5, 4))
    e = rng.uniform(50, 500, n)
    m[:, 0] = np.stack([e, 0 * e, 0 * e, e], 1)
    m[:, 1] = np.stack([e, 0 * e, 0 * e, -e], 1)
    m[:, 2:, 0] = 2 * e[:, None] / 3
    saved = mg.label_lhe_pp
    mg.label_lhe_pp = lambda process, mom, a, sa=None: np.asarray(a, float)      # the label is the alpha_s itself
    try:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, f"{NAME}_test.npy")
            rec = {"role": "test", "lhe": {"block": [0, n]}}
            mg._lhe_finish(NAME, [rec], [(rec, path)], [m], [np.full(n, 0.118)], [(1, n, {})], None, {}, None, 42, 1,
                           n, 1, False)
            a = np.load(path[:-len(".npy")] + ".alphas.npy")
            rows = np.load(path)[:, :20].reshape(-1, 5, 4)          # the pool's rows, shuffled as written
            q = rows[:, 0] + rows[:, 1]
            sq = np.sqrt(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1))
    finally:
        mg.label_lhe_pp = saved
        mg.PROCESSES.pop(NAME, None)
    return sq, a


def test_label_alphas_mz():
    sq, a = _finish({"label_alphas_mz": 0.13})
    assert np.allclose(a, mg.compute_alphas(sq, alphas_mz=0.13))
    sq0, a0 = _finish({})
    assert np.allclose(a0, mg.compute_alphas(sq0, alphas_mz=0.118))
    assert np.all(np.sort(a) > np.sort(a0))
