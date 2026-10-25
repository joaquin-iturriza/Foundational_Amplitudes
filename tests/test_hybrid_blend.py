"""data.target_physics_hybrid_blend (physics_scales.hybrid_blend, added in experiment._physics_logF): the hybrid
target plus lambda times the per-event emission-scale term minus its process mean. lambda = 0 is the hybrid target,
lambda = 1 the emission-scale target (data.target_physics_dim_ref=qperp): the constant c_p cancels there, whichever
events it was fitted on. CPU only, numpy, no torch.

Run as a script (pytest is shadowed by the stray root-level py.py):
    python tests/test_hybrid_blend.py
"""
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from physics_scales import hybrid_blend, hybrid_cp, qperp_logs  # noqa: E402
from test_hybrid_cp_frozen import _events  # noqa: E402

LS0 = 0.0   # log s0 with target_physics_s0 = 1, as the short job's arms


def _log_shat(P):
    q = P[:, 0] + P[:, 1]
    return np.log(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(axis=1))


def _hybrid_logF(P, cp, lam=0.0):
    """experiment._physics_logF at dim_ref=shat, dim_only, target_physics_hybrid (coupling and order terms left out:
    they enter every arm alike)."""
    N = P.shape[1]
    return -((4 - N) * (_log_shat(P) - LS0)) + cp + hybrid_blend(P, cp, lam)


def _qperp_logF(P):
    """experiment._physics_logF at dim_ref=qperp, dim_only."""
    return qperp_logs(P) - (P.shape[1] - 4) * LS0


def test_lambda0_is_the_hybrid():
    P = _events(3, seed=5)
    cp = hybrid_cp(P)
    assert np.array_equal(hybrid_blend(P, cp, 0.0), np.zeros(len(P)))
    assert np.allclose(_hybrid_logF(P, cp, 0.0), _hybrid_logF(P, cp))


def test_lambda1_is_the_emission_scale_target():
    for nf, seed in ((3, 6), (4, 7)):
        P = _events(nf, seed=seed)
        cp = hybrid_cp(P)                                   # fitted on the same events: exact
        assert np.allclose(_hybrid_logF(P, cp, 1.0), _qperp_logF(P), atol=1e-9)
        # c_p from other events (a frozen parent's): the constant cancels at lambda = 1, still exact
        cp_other = hybrid_cp(_events(nf, seed=seed + 100))
        assert np.allclose(_hybrid_logF(P, cp_other, 1.0), _qperp_logF(P), atol=1e-9)


def test_blend_interpolates_and_keeps_the_class_mean():
    P = _events(4, seed=8)
    cp = hybrid_cp(P)
    h, q = _hybrid_logF(P, cp, 0.0), _qperp_logF(P)
    for lam in (0.25, 0.5):
        b = _hybrid_logF(P, cp, lam)
        assert np.allclose(b, (1 - lam) * h + lam * q, atol=1e-9)
        assert abs(b.mean() - h.mean()) < 1e-9              # the term has zero mean over the fitting events
    assert np.array_equal(hybrid_blend(_events(2), 0.0, 0.5), np.zeros(500))   # 2 -> 2: nothing to blend


def test_experiment_wiring():
    src = open(os.path.join(ROOT, "experiment.py")).read()
    assert re.search(r'lcp = lcp \+ hybrid_blend\(mom_phys, self\._phys_cp\[name\], _lam\)', src)
    assert '"target_physics_hybrid_blend": float(self.cfg.data.get("target_physics_hybrid_blend", 0.0))' in src
    assert '("target_physics_hybrid_blend", 0.0)' in src     # asserted on frozen stats


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
