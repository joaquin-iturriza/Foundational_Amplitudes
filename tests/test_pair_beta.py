"""experiment.pair_beta: the pair's rest-frame velocity, boost invariant, down to beta 1e-5 (data.target_times_beta)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_pair_beta():
    from experiment import pair_beta
    m, beta = 172.5, np.array([1e-5, 1e-3, 0.2, 0.6])
    e = m / np.sqrt(1 - beta ** 2)
    pz = e * beta
    cm = np.zeros((len(beta), 4, 4))
    cm[:, 0] = np.stack([e, 0 * e, 0 * e, e], 1)            # massless beams
    cm[:, 1] = np.stack([e, 0 * e, 0 * e, -e], 1)
    cm[:, 2] = np.stack([e, pz * 0.6, pz * 0.8, 0 * e], 1)  # t, then t~
    cm[:, 3] = np.stack([e, -pz * 0.6, -pz * 0.8, 0 * e], 1)
    g, v = 1 / np.sqrt(1 - 0.5 ** 2), 0.5                   # a boost along z
    lab = cm.copy()
    lab[..., 0], lab[..., 3] = g * (cm[..., 0] + v * cm[..., 3]), g * (cm[..., 3] + v * cm[..., 0])
    ids = np.tile([11, -11, -6, 6], (len(beta), 1))
    for mom in (cm, lab):
        np.testing.assert_allclose(pair_beta(mom.reshape(len(beta), -1), ids), beta, rtol=1e-5)


if __name__ == "__main__":
    test_pair_beta(); print("ok")
