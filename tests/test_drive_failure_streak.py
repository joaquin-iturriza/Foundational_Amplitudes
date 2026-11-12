"""sweep/drive.py failure_streak: the driver stops when the last N finished trials failed with one cause."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep.drive import failure_streak  # noqa: E402

BORN = "AssertionError: data.target_over_born: x.born.npy was computed on another pool than x.npy"


def tr(i, status, cause=None):
    return {"idx": i, "finished": 100 + i, "status": status, "cause": cause}


def test_same_crash_stops():
    assert failure_streak([tr(i, "failed", BORN) for i in range(6)], 6) == BORN


def test_an_observed_trial_or_other_cause_breaks_the_streak():
    assert failure_streak([tr(i, "failed", BORN) for i in range(5)] + [tr(5, "observed")], 6) is None
    assert failure_streak([tr(i, "failed", BORN) for i in range(5)] + [tr(5, "failed", "ValueError: y")], 6) is None
    assert failure_streak([tr(i, "failed", BORN) for i in range(5)], 6) is None          # under N finished


def test_divergence_and_off():
    assert failure_streak([tr(i, "failed", "RuntimeError: Training diverged at step N") for i in range(6)], 6) is None
    assert failure_streak([tr(i, "failed", BORN) for i in range(6)], 0) is None


if __name__ == "__main__":
    for f in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        f()
    print("ok")
