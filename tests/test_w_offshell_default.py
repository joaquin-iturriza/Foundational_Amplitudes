"""D506: the W (24) is in the default internal_mass_pdgs, and init_data refuses a new off-shell training run without it."""
import os

import pytest
from omegaconf import OmegaConf

from experiment import AmplitudeExperiment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _init_data(pdgs, **top):
    exp = AmplitudeExperiment.__new__(AmplitudeExperiment)
    exp.cfg = OmegaConf.create(dict({"data": {"internal_mass_scalars": True, "offshell_per_event": True,
                                              "internal_mass_pdgs": pdgs}, "train": True}, **top))
    try:
        exp.init_data()
    except Exception as e:   # past the check, init_data fails on the missing rest of the config
        return str(e)
    return ""


def test_default_has_w():
    cfg = OmegaConf.load(os.path.join(ROOT, "config", "amplitudes.yaml"))
    assert list(cfg.data.internal_mass_pdgs) == [23, 6, 25, 24]


def test_refuses_new_run_without_w():
    assert "D506" in _init_data([23, 6, 25])


@pytest.mark.parametrize("top", [{}, {"fine_tune": {"pretrained_path": "x"}}, {"warm_start_idx": 0},
                                 {"train": False}])
def test_passes_with_w_or_inherited_columns(top):
    pdgs = [23, 6, 25, 24] if not top else [23, 6, 25]
    assert "D506" not in _init_data(pdgs, **top)
