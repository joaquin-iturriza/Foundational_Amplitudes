"""training.save_last: the last model is skipped only when it is false and a best checkpoint is kept (base_experiment
._save_last_model); the default config keeps saving it."""
import os, sys

from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from base_experiment import BaseExperiment  # noqa: E402


def _saves(training):
    exp = object.__new__(BaseExperiment)
    exp.cfg = OmegaConf.create({"training": training})
    calls = []
    exp._save_model = lambda *a, **k: calls.append(a)
    exp._save_last_model()
    return len(calls)


def test_save_last():
    assert OmegaConf.load(os.path.join(ROOT, "config", "default.yaml")).training.save_last is True
    assert _saves({"save_last": True, "es_load_best_model": True}) == 1
    assert _saves({"es_load_best_model": True}) == 1                       # key absent (older configs): saved
    assert _saves({"save_last": False, "es_load_best_model": True}) == 0
    assert _saves({"save_last": False, "es_load_best_model": False}) == 1  # no best checkpoint: keep the only model


if __name__ == "__main__":
    test_save_last(); print("OK")
