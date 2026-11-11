"""keyed_aug.pool_identity, the memmap cache key's pool identity (data_mmap format 3): a pool composed at load is
identified by its rows' digest, a stored pool by its recipe_id, size and the bytes at both ends; neither by its path
or mtime, so every job of the same data opens one cache wherever its pools lie."""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import json, os, shutil, sys, tempfile, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keyed_aug import pool_identity


def _composed(d, rows, sha="ab" * 32, asha="cd" * 32):
    p = os.path.join(d, "v4_x_train_smixd25_amplitudes.npy")
    np.save(p, rows); np.save(p[:-4] + ".alphas.npy", rows[:, :1])
    json.dump({"recipe_id": "r1", "derived_from": {"composed_at_load": True, "rows_sha256": sha,
                                                   "alphas_sha256": asha}}, open(p + ".recipe.json", "w"))
    return p


def test_composed_pool_identity_stable_across_jobs():
    rows = np.arange(12.0).reshape(4, 3)
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        pa = _composed(a, rows)
        time.sleep(0.01)
        pb = _composed(b, rows)
        assert pool_identity(pa) == pool_identity(pb) == ["r1", os.path.getsize(pa), "rows:" + "ab" * 32]
        assert pool_identity(pa[:-4] + ".alphas.npy") == pool_identity(pb[:-4] + ".alphas.npy")
        pc = _composed(b, rows, sha="ef" * 32)          # other content, other identity
        assert pool_identity(pc) != pool_identity(pa)


def test_stored_pool_identity_is_content():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        p = os.path.join(a, "pool.npy")
        np.save(p, np.random.default_rng(0).normal(size=(50000, 9)))
        json.dump({"recipe_id": "rid"}, open(p + ".recipe.json", "w"))
        q = os.path.join(b, "pool.npy")
        shutil.copy(p, q); shutil.copy(p + ".recipe.json", q + ".recipe.json")
        time.sleep(0.01); os.utime(q, None)
        assert pool_identity(p) == pool_identity(q) and pool_identity(p)[0] == "rid"
        x = np.load(q); x[-1, -1] += 1; np.save(q, x)                # the last row changed
        assert pool_identity(p) != pool_identity(q)
        x = np.load(p)[:-1]; np.save(q, x)                           # one row fewer (the header's shape)
        assert pool_identity(p) != pool_identity(q)


if __name__ == "__main__":
    test_composed_pool_identity_stable_across_jobs(); test_stored_pool_identity_is_content(); print("ok")
