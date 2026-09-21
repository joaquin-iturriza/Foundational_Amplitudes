"""AmplitudeDataset.__getitems__ (one gather per batch) against the per-item path.

CPU only. Run as a script (pytest is shadowed by the stray root-level py.py):
    python tests/test_batch_fetch.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import AmplitudeDataset, build_flat_arrays, collate_variable_length  # noqa: E402


def _dataset(n_events=300, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.integers(4, 8, size=n_events)
    particles = [rng.normal(size=(c, 4)).astype(np.float32) for c in counts]
    tokens = [rng.integers(1, 20, size=c) for c in counts]
    flat_p, flat_t, offsets = build_flat_arrays(particles, tokens)
    return AmplitudeDataset(
        particles_flat=flat_p, offsets=offsets,
        amplitudes=rng.normal(size=(n_events, 1)).astype(np.float32),
        tokens_flat=flat_t,
        order_labels=rng.normal(size=(n_events, 4)).astype(np.float32),
        dtype=torch.float32,
        process_ids=rng.integers(0, 5, size=n_events),
    )


def _both(ds, idx):
    batched = collate_variable_length(ds.__getitems__(idx))
    items = collate_variable_length([ds[i] for i in idx])
    return batched, items


def test_batch_fetch_matches_items():
    ds = _dataset()
    rng = np.random.default_rng(1)
    for idx in (list(range(10)), list(rng.permutation(300)[:64]), [299], [5, 5, 7]):
        batched, items = _both(ds, idx)
        assert len(batched) == len(items) == 6
        for a, b in zip(batched, items):
            assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, b.dtype, a.shape, b.shape)
            assert torch.equal(a, b)


def test_dataloader_uses_the_batch_path():
    ds = _dataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, drop_last=False,
                                     collate_fn=collate_variable_length, num_workers=0)
    n = 0
    for batch in dl:
        particles, amplitudes, tokens, order_labels, ptr, pids = batch
        assert ptr[-1] == particles.shape[0] == tokens.shape[0]
        assert amplitudes.shape[0] == order_labels.shape[0] == pids.shape[0] == ptr.shape[0] - 1
        n += amplitudes.shape[0]
    assert n == len(ds)


def test_item_toggle():
    os.environ["LLOCA_FETCH"] = "item"
    try:
        ds = _dataset()
        out = ds.__getitems__([1, 2, 3])
        assert len(out) == 3 and isinstance(out[0], tuple)
    finally:
        del os.environ["LLOCA_FETCH"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
