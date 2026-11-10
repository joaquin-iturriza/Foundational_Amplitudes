"""CPU checks of training.is_weight_path (experiment._is_weighted, _get_is_weights; port of l2-gen 5fe8545d): with the
path null the per-event loss is unchanged; with a table each training event's loss is multiplied by c_b of its raw
log|M|^2 bin (the standardized target mapped back through the pool's amplitude preprocessing); in eval mode nothing
is weighted (validation stays the plain mean); the heterosc_mu_only and HETEROSC paths are weighted too; a run with
more than one amplitude preprocessing is refused; and a pool drawn from pi with c_b = Q_b/pi_b gives the Q-weighted
mean loss in expectation (unbiased IS).
    .venv-cpu/bin/python -m pytest tests/test_is_weight_path.py
"""
import json
import os
import sys
import tempfile
import types

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiment import AmplitudeExperiment as E  # noqa: E402

EDGES = [-10.0, -5.0, 0.0, 5.0, 10.0]          # 4 log|M|^2 bins
C = [4.0, 0.5, 1.0, 2.0]
AMP_MEAN, AMP_STD = 1.0, 2.0


def fake(path, loss="MSE", training=True, prepd_mean=(AMP_MEAN,), **extra):
    cfg = OmegaConf.create({"training": dict(loss=loss, is_weight_path=path, **extra)})
    ex = types.SimpleNamespace(cfg=cfg, device=torch.device("cpu"), prepd_mean=list(prepd_mean),
                               prepd_std=[AMP_STD] * len(prepd_mean),
                               model=types.SimpleNamespace(training=training))
    for name in ("_per_event_loss", "_is_weighted", "_get_is_weights"):
        setattr(ex, name, types.MethodType(getattr(E, name), ex))
    return ex


@pytest.fixture
def table():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "is_weights.json")
        with open(p, "w") as f:
            json.dump({"logm_edges": EDGES, "c": C}, f)
        yield p


def targets():
    logm = torch.tensor([-7.0, -2.0, 3.0, 8.0, -12.0, 15.0], dtype=torch.float64)   # bins 0,1,2,3, clamped 0 and 3
    return ((logm - AMP_MEAN) / AMP_STD)[:, None], torch.tensor([4.0, 0.5, 1.0, 2.0, 4.0, 2.0], dtype=torch.float64)


def test_null_path_is_identity():
    y, _ = targets()
    yp = y + 0.3
    out = fake(None)._per_event_loss(yp, y)
    assert torch.allclose(out, ((yp - y) ** 2)[:, 0])


def test_training_weights_by_bin(table):
    y, c = targets()
    yp = y + 0.3
    out = fake(table)._per_event_loss(yp, y)
    assert torch.allclose(out, ((yp - y) ** 2)[:, 0] * c)


def test_eval_mode_not_weighted(table):
    y, _ = targets()
    yp = y + 0.3
    out = fake(table, training=False)._per_event_loss(yp, y)
    assert torch.allclose(out, ((yp - y) ** 2)[:, 0])


def test_heterosc_and_mu_only_weighted(table):
    y, c = targets()
    yp, sigma = y + 0.3, torch.full_like(y, 0.5)
    plain = fake(None, loss="HETEROSC")._per_event_loss(yp, y, sigma=sigma)
    assert torch.allclose(fake(table, loss="HETEROSC")._per_event_loss(yp, y, sigma=sigma), plain * c)
    mu = fake(table, loss="HETEROSC", heterosc_mu_only=True)._per_event_loss(yp, y, sigma=sigma)
    assert torch.allclose(mu, ((yp - y) ** 2)[:, 0] * c)


def test_multi_process_refused(table):
    y, _ = targets()
    with pytest.raises(ValueError):
        fake(table, prepd_mean=(1.0, 2.0))._per_event_loss(y, y)


def test_unbiased_estimator(table):
    rng = np.random.default_rng(0)
    Q = np.array([0.25, 0.25, 0.25, 0.25])          # objective marginal over the bins
    pi = np.array([0.0625, 0.5, 0.25, 0.1875])      # emphasized pool density
    c = Q / pi
    with open(table, "w") as f:
        json.dump({"logm_edges": EDGES, "c": c.tolist()}, f)
    centres = np.array([-7.5, -2.5, 2.5, 7.5])
    err = np.array([1.0, 0.1, 0.2, 3.0])            # per-bin squared error
    b = rng.choice(4, size=200_000, p=pi)
    y = torch.tensor((centres[b] - AMP_MEAN) / AMP_STD)[:, None]
    yp = y + torch.tensor(np.sqrt(err[b]))[:, None]
    weighted = float(fake(table)._per_event_loss(yp, y).mean())
    assert abs(weighted - float((Q * err).sum())) < 0.02 * float((Q * err).sum())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
