"""sqrts_measure log_beta (mg5_pipeline_final._sqrts_sampler): kit_benchmark's threshold pools, a 2->2 top pair with
beta = sqrt(1 - 4 m^2 / s) log-uniform on [1e-5, 0.6], drawn plainly (sample_2to2_phase_space)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mg5_pipeline_final as mg  # noqa: E402

MT = 172.5
LO, HI = (2 * MT / np.sqrt(1 - b * b) for b in (1e-5, 0.6))


def _beta(P):
    s = (P[:, 0, 0] + P[:, 1, 0]) ** 2 - (P[:, 0, 3] + P[:, 1, 3]) ** 2
    return np.sqrt(1 - 4 * MT ** 2 / s), s


def test_beta_log_uniform_in_range_and_on_shell():
    ev, sq = mg.sample_2to2_phase_space(20000, LO, HI, MT, [21, 21, 6, -6], rng=np.random.default_rng(1),
                                        sqrts_measure="log_beta", beam_mass={})
    P = np.stack([e[0] for e in ev])
    beta, s = _beta(P)
    assert np.allclose(np.sqrt(s), sq, rtol=1e-12)
    assert beta.min() >= 1e-5 * (1 - 1e-4) and beta.max() <= 0.6 * (1 + 1e-12)
    assert beta.min() < 2e-5 and beta.max() > 0.5            # both ends reached
    u = (np.log(beta) - np.log(1e-5)) / (np.log(0.6) - np.log(1e-5))   # uniform on [0, 1] if log-uniform
    assert np.abs(np.quantile(u, [0.1, 0.25, 0.5, 0.75, 0.9]) - [0.1, 0.25, 0.5, 0.75, 0.9]).max() < 0.02
    m2 = P[:, 2:, 0] ** 2 - (P[:, 2:, 1:] ** 2).sum(-1)
    assert np.allclose(m2, MT ** 2, rtol=1e-6)               # tops on shell
    assert np.allclose(P[:, :2, 0] ** 2 - (P[:, :2, 1:] ** 2).sum(-1), 0, atol=1e-6)   # massless beams


def test_uniform_stream_unchanged():
    """the default measure draws sqrt(s) exactly as before the measure argument (rng.uniform on the window)"""
    ev, sq = mg.sample_2to2_phase_space(100, 400.0, 1000.0, MT, [11, -11, 6, -6], rng=np.random.default_rng(3))
    assert np.array_equal(sq, np.random.default_rng(3).uniform(400.0, 1000.0, 100))


def test_entry_level_measure_registers_under_own_name():
    if "v4_gg_ttx" not in mg.PROCESSES:
        mg.PROCESSES["v4_gg_ttx"] = {"pdg_ids": [21, 21, 6, -6], "m_final": MT}
    mg.register_recipe_processes([{"name": "v4_gg_ttx__thr", "base": "v4_gg_ttx", "sqrts_measure": "log_beta",
                                   "sampling": {"mode": "flat"}}])
    assert mg.PROCESSES["v4_gg_ttx__thr"]["sqrts_measure"] == "log_beta"
    assert mg.PROCESSES["v4_gg_ttx"].get("sqrts_measure") != "log_beta"
    assert mg.sampling_policy(mg.PROCESSES["v4_gg_ttx__thr"])["mode"] == "flat"


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
