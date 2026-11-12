"""datagen.finalize_dataset writes a one-loop pool's Born identity record after its .recipe.json, so
experiment._born_of_pool accepts the pool it just built (repo-reviewer 2026-11-12: written before, it held a null
recipe_id and every V/B run on a new pool was refused)."""
import _tmpdir  # noqa: F401
import os, sys, tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_finalize_writes_a_matching_born_record():
    pytest.importorskip("torch")
    import datagen
    from experiment import AmplitudeExperiment
    proc = "v4_bbx_zz_nlo"
    assert datagen.is_virt(proc)
    with tempfile.TemporaryDirectory() as d:
        work, dest = os.path.join(d, "work"), os.path.join(d, "dest")
        os.makedirs(work)
        chunks = []
        for i in range(2):
            p = os.path.join(work, f"c{i}.npy")
            np.save(p, np.random.default_rng(i).random((3, 21)) + 1)
            np.save(p[:-4] + ".born.npy", np.full(3, 2.0 + i))
            chunks.append(p)
        final = datagen.finalize_dataset(proc, 200.0, 6000.0, 6, "train", 0, chunks, dest, work)
        b = AmplitudeExperiment._born_of_pool(final, 6)
        assert np.allclose(b[:, 0], [2, 2, 2, 3, 3, 3])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
