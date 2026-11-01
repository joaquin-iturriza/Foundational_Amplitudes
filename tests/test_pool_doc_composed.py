"""data_mmap._pool_doc: a pool composed at load is keyed by its content digest (not its node-local path or mtime), so
every job of the same data opens one memmap cache; a stored pool keeps path, size and mtime."""
import json, os, sys, tempfile, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_mmap


def _composed(d, rows, sha="ab" * 32, asha="cd" * 32):
    p = os.path.join(d, "v4_x_train_smixd25_amplitudes.npy")
    np.save(p, rows); np.save(p[:-4] + ".alphas.npy", rows[:, :1])
    json.dump({"recipe_id": "r1", "derived_from": {"composed_at_load": True, "rows_sha256": sha,
                                                   "alphas_sha256": asha}}, open(p + ".recipe.json", "w"))
    return p


def test_composed_pool_doc_stable_across_jobs():
    rows = np.arange(12.0).reshape(4, 3)
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        pa = _composed(a, rows)
        time.sleep(0.01)
        pb = _composed(b, rows)
        assert data_mmap._pool_doc(pa) == data_mmap._pool_doc(pb) == [os.path.basename(pa), os.path.getsize(pa),
                                                                      "sha256:" + "ab" * 32, "r1"]
        aa, ab = pa[:-4] + ".alphas.npy", pb[:-4] + ".alphas.npy"
        assert data_mmap._pool_doc(aa) == data_mmap._pool_doc(ab)
        assert data_mmap._pool_doc(aa)[2] == "sha256:" + "cd" * 32
        pc = _composed(b, rows, sha="ef" * 32)          # other content, other key
        assert data_mmap._pool_doc(pc) != data_mmap._pool_doc(pa)


def test_stored_pool_doc_keeps_mtime():
    with tempfile.TemporaryDirectory() as a:
        p = os.path.join(a, "v4_y_train_amplitudes.npy")
        np.save(p, np.zeros((2, 2)))
        json.dump({"recipe_id": "r2"}, open(p + ".recipe.json", "w"))
        d = data_mmap._pool_doc(p)
        assert d == [os.path.abspath(p), os.path.getsize(p), os.stat(p).st_mtime_ns, "r2"]
        al = p[:-4] + ".alphas.npy"; np.save(al, np.zeros(2))
        assert data_mmap._pool_doc(al)[3] is None and data_mmap._pool_doc(al)[0] == os.path.abspath(al)


if __name__ == "__main__":
    test_composed_pool_doc_stable_across_jobs(); test_stored_pool_doc_keeps_mtime(); print("ok")
