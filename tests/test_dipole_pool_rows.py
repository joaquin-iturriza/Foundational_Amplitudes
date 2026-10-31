"""dipole_residual.pool_rows: a read's idx (positions in the role-contiguous train|val|test array, each role shuffled by
one rng seeded 42 as experiment._init_data does) map back to the pool rows the events came from."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.catalog_v4.dipole_residual import pool_rows  # noqa: E402


class Z(dict):
    @property
    def files(self):
        return list(self)


def _layout(counts):
    rng = np.random.default_rng(seed=42)
    return [rng.permutation(c) for c in counts]


def test_val_and_test_rows():
    nt, nv, ns = 10000, 2000, 2000
    pv, ps = _layout([nt, nv, ns])[1:]
    for split, perm, start in (("val", pv, nt), ("test", ps, nt + nv)):
        idx = np.arange(start, start + len(perm))           # scored in loader order
        rows = pool_rows(Z(idx=idx, split=np.array(split)))
        assert np.array_equal(rows, perm)                   # position i of the role holds pool row perm[i]


def test_saved_pool_row_wins():
    z = Z(idx=np.arange(10, 13), split=np.array("val"), pool_row=np.array([2, 0, 1]))
    assert list(pool_rows(z)) == [2, 0, 1]


if __name__ == "__main__":
    test_val_and_test_rows()
    test_saved_pool_row_wins()
    print("ok")


def test_eta2_and_tail_share():
    """eta2 sees a mean that depends on x with no linear tilt; tail_share sees an error concentrated at one end."""
    from analysis.catalog_v4.dipole_residual import eta2, tail_share
    r = np.random.default_rng(0)
    x = r.normal(size=4000)
    u = np.sin(3 * x) + 0.3 * r.normal(size=4000)
    assert eta2(u, x) > 0.6 and np.corrcoef(u, x)[0, 1] ** 2 < 0.05
    v = np.where(x > 1.3, 3.0, 0.0) * r.normal(size=4000) + 0.1 * r.normal(size=4000)
    lo, hi = tail_share(v, x)
    assert hi > 0.9 and lo < 0.05
    w = r.normal(size=4000)
    assert eta2(w, x) < 0.02 and all(0.07 < s < 0.13 for s in tail_share(w, x))


def test_joint_r2_finds_signal_and_not_noise():
    """The joint read's out-of-fold fit recovers a nonlinear signal in one feature and reads no signal in noise."""
    from analysis.catalog_v4.dipole_residual import joint_features, joint_r2
    r = np.random.default_rng(1)
    m = r.normal(size=(2000, 6, 4))
    m[..., 0] = np.abs(m[..., 0]) + 3
    t = r.normal(size=2000)
    X = joint_features(m, t, t * 0, t * 0, None)
    u = np.sin(2 * t) + 0.5 * r.normal(size=2000)
    rj, rn = joint_r2(X, u)
    assert rj > 0.5 and rn < 0.05
    assert joint_r2(X, r.normal(size=2000))[0] < 0.05
