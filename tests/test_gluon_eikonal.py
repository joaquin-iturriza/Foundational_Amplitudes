"""experiment.gluon_eikonal: prod over final quarks of 2 p_g.p_q / s, boost invariant (data.target_times_eikonal)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _massless(rng, n):
    v = rng.normal(size=(n, 3))
    return np.concatenate([np.linalg.norm(v, axis=1, keepdims=True), v], 1)


def test_gluon_eikonal():
    from experiment import gluon_eikonal
    rng = np.random.default_rng(0)
    n, E = 5, 500.0
    beams = np.zeros((n, 2, 4))
    beams[:, 0] = [E, 0, 0, E]
    beams[:, 1] = [E, 0, 0, -E]
    fin = np.stack([_massless(rng, n) * 100, _massless(rng, n) * 100, _massless(rng, n) * 1e-3], 1)   # u, u~, soft g
    mom = np.concatenate([beams, fin], 1)
    ids = np.tile([11, -11, 2, -2, 21], (n, 1))
    dot = lambda a, b: a[:, 0] * b[:, 0] - (a[:, 1:] * b[:, 1:]).sum(1)
    s = 4 * E ** 2
    want = 2 * dot(mom[:, 4], mom[:, 2]) / s * 2 * dot(mom[:, 4], mom[:, 3]) / s
    np.testing.assert_allclose(gluon_eikonal(mom.reshape(n, -1), ids), want, rtol=1e-12)
    g, v = 1 / np.sqrt(1 - 0.5 ** 2), 0.5                   # a boost along z leaves it unchanged
    lab = mom.copy()
    lab[..., 0], lab[..., 3] = g * (mom[..., 0] + v * mom[..., 3]), g * (mom[..., 3] + v * mom[..., 0])
    np.testing.assert_allclose(gluon_eikonal(lab.reshape(n, -1), ids), want, rtol=1e-9)
    with pytest.raises(AssertionError):                     # two final gluons: not this factor
        gluon_eikonal(np.concatenate([mom, mom[:, 4:5]], 1).reshape(n, -1), np.tile([11, -11, 2, -2, 21, 21], (n, 1)))


if __name__ == "__main__":
    test_gluon_eikonal(); print("ok")
