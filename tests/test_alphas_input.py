"""data.alphas_input on the memmap path (data_mmap.MmapSource.gather): each event's alpha_s as one more order column.

CPU only: python tests/test_alphas_input.py
Checks, on a two-process cache written here: the column is (ln alpha_s + 1.994) / 0.169 of each event in the virtual
row order, after the order row and before dimensionless_inputs' column; with the flag off the order rows are
unchanged. The wiring in init_data_recipes_mmap (n_order_features + 1, the refusal at load of any read event without
alpha_s) is exercised by the GPU smoke run.
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import data_mmap as dm  # noqa: E402


def _source(d, alphas, dimless=False):
    n, P = len(alphas), 2
    rng = np.random.default_rng(0)
    np.save(os.path.join(d, "mom.npy"), rng.normal(size=(n * P, 4)).astype(np.float64))
    np.save(os.path.join(d, "pdg.npy"), np.full(n * P, 21, dtype=np.int64))
    np.save(os.path.join(d, "rs.npy"), rng.uniform(100, 1000, n))
    np.save(os.path.join(d, "evp.npy"), np.arange(n, dtype=np.int64) * P)
    np.save(os.path.join(d, "alphas.npy"), np.asarray(alphas, dtype=np.float64))
    meta = {"segs": [{"ev0": 0, "P": P, "uid": 1}, {"ev0": n // 2, "P": P, "uid": 2}]}
    ev_rows = rng.permutation(n)                     # the run's shuffled order: virtual row -> base row
    order_rows = np.array([[1, 0, 2, 0], [0, 0, 0, 3]], dtype=np.float32)
    src = dm.MmapSource(d, d, meta, ev_rows, 42, 1.0, dimless, np.array([21]), np.zeros((2, 1), dtype=np.int64),
                        order_rows, None, 1e-6, 4)
    return src, ev_rows, order_rows


def _turn_on(src, d):
    src.alp = dm._open(d, "alphas")
    src.alp_shift, src.alp_scale = 1.994, 0.169


def main():
    with tempfile.TemporaryDirectory() as d:
        a = np.linspace(0.08, 0.2, 10)
        src, ev_rows, order_rows = _source(d, a)
        v = np.arange(10)
        pid = (ev_rows[v] >= 5).astype(np.int64)
        _, _, off, _ = src.gather(v, pid)
        assert np.array_equal(off, order_rows[pid]), "flag off must leave the order rows unchanged"

        _turn_on(src, d)
        _, _, on, _ = src.gather(v, pid)
        assert on.shape == (10, 5) and np.array_equal(on[:, :4], off)
        np.testing.assert_allclose(on[:, 4], (np.log(a[ev_rows]) + 1.994) / 0.169, rtol=1e-6)

    with tempfile.TemporaryDirectory() as d:         # dimensionless_inputs' column stays last
        a = np.linspace(0.08, 0.2, 10)
        src, ev_rows, _ = _source(d, a, dimless=True)
        _turn_on(src, d)
        v = np.arange(10)
        _, _, on, _ = src.gather(v, (ev_rows[v] >= 5).astype(np.int64))
        np.testing.assert_allclose(on[:, 4], (np.log(a[ev_rows]) + 1.994) / 0.169, rtol=1e-6)
        np.testing.assert_allclose(on[:, 5], np.log10(np.load(os.path.join(d, "rs.npy"))[ev_rows]), rtol=1e-6)
    print("OK")


if __name__ == "__main__":
    main()
