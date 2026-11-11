"""plan virtual_over_born: the amplitude std floor (preprocessing.AMP_STD_FLOOR) and the Born sidecar read of
data.target_over_born (AmplitudeExperiment._born_of_pool)."""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import os, sys, tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import preprocess_amplitude, undo_preprocess_amplitude, AMP_STD_FLOOR


def test_constant_target_is_not_blown_up():
    # e+e- -> u u~ over its Born: C_F (pi^2 - 8) up to rounding (the June datasets: relative std 5e-13)
    rng = np.random.default_rng(0)
    x = (2.4928058681 * (1 + 5e-13 * rng.standard_normal(70000)))[:, None]
    y, m, s = preprocess_amplitude(x, trafos=["log", "standardization"])
    assert s == AMP_STD_FLOOR
    assert np.abs(y).max() < 1e-5
    back = undo_preprocess_amplitude(y, m, s, trafos=["log", "standardization"])
    assert np.allclose(back, x, rtol=1e-12)


def test_ordinary_target_unchanged():
    rng = np.random.default_rng(1)
    x = np.exp(rng.standard_normal((5000, 1)) * 2)
    y, m, s = preprocess_amplitude(x, trafos=["log", "standardization"])
    lx = np.log(x)
    assert np.allclose(y, (lx - lx.mean(0)) / lx.std(0)) and np.isclose(s, lx.std(0)).all()


def test_born_of_pool():
    pytest.importorskip("torch")
    from experiment import AmplitudeExperiment
    with tempfile.TemporaryDirectory() as d:
        pool = os.path.join(d, "p_train_amplitudes.npy")
        np.save(pool, np.zeros((4, 21)))
        with pytest.raises(AssertionError):
            AmplitudeExperiment._born_of_pool(pool, 4)
        import json
        from keyed_aug import pool_identity
        np.save(pool[:-4] + ".born.npy", np.array([1.0, 2.0, 4.0, 8.0]))
        with pytest.raises(AssertionError):          # no identity record
            AmplitudeExperiment._born_of_pool(pool, 3)
        json.dump({"pool_identity": pool_identity(pool), "n_rows": 4}, open(pool[:-4] + ".born.npy.json", "w"))
        b = AmplitudeExperiment._born_of_pool(pool, 3)
        assert b.shape == (3, 1) and np.allclose(b[:, 0], [1, 2, 4])
        np.save(pool[:-4] + ".born.npy", np.array([1.0, 0.0, 4.0, 8.0]))
        with pytest.raises(AssertionError):
            AmplitudeExperiment._born_of_pool(pool, 4)
        np.save(pool[:-4] + ".born.npy", np.array([1.0, 2.0, 4.0, 8.0]))
        np.save(pool, np.ones((4, 21)))              # the pool regenerated under the surviving sidecar
        with pytest.raises(AssertionError):
            AmplitudeExperiment._born_of_pool(pool, 4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
