"""analysis/catalog_v4/dipole_residual.pool_rows: a read's split positions map back to the pool rows they came from.
    .venv-cpu/bin/python -m pytest -q tests/test_dipole_pool_rows.py"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.catalog_v4.dipole_residual import pool_rows  # noqa: E402


def _shuffle(n):
    rng = np.random.default_rng(seed=42)
    pools = {r: np.arange(k) * 10 + i for i, (r, k) in enumerate(zip(("train", "val", "test"), n))}
    return np.concatenate([pools[r][rng.permutation(k)] for r, k in zip(("train", "val", "test"), n)])


def test_positions_map_to_pool_rows():
    n = (50, 13, 7)
    arr = _shuffle(n)       # value = 10 * pool row + role code
    for role, start, code in (("val", 50, 1), ("test", 63, 2), ("train", 0, 0)):
        idx = np.arange(start, start + n[("train", "val", "test").index(role)])[::-1][:5]
        rows = pool_rows({"idx": idx, "split": np.array(role), "role_counts": np.array(n)})
        assert np.array_equal(arr[idx], rows * 10 + code)


def test_saved_pool_row_wins():
    z = {"idx": np.array([50]), "split": np.array("val"), "role_counts": np.array([50, 13, 7]),
         "pool_row": np.array([3])}
    assert pool_rows(z).tolist() == [3]
