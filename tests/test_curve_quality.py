"""curve_quality.quality: a run is good only when validation goes down together with training (Joaquin 2026-11-12):
validation that flattens while training keeps falling fails, as does validation that turns up; validation that
follows training, even jittery, passes."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "data_saving"))
from curve_quality import quality


def test_together():
    n = 40
    train = np.logspace(0, -4, n)
    assert quality(train * 2 * (1 + 0.8 * (np.arange(n) % 2)), train)["well"]         # together, 2x jitter
    flat = np.r_[np.logspace(0, -2, 20), np.full(20, 1e-2)]                            # val flat, train falls 2 decades
    q = quality(flat, train)
    assert not q["well"] and q["separation"] > 30 and q["end_up"] < 2
    slow = np.logspace(0, -3, n)                                                        # val falls 3 decades of train's 4
    assert not quality(slow, train)["well"]
    up = np.r_[train[:30], np.logspace(-3, -2, 10)]                                     # turns up at the end
    q = quality(up, train)
    assert not q["well"] and q["end_up"] > 2


if __name__ == "__main__":
    test_together()
