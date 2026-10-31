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
