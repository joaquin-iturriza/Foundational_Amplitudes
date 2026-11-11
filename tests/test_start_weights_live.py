"""CPU check of fine_tune.start_weights (plan warm_start step 14): with "live" the fine-tune loads the parent
checkpoint's live training weights (model_live), with the default "scored" its EMA weights (model, D63 format); a
checkpoint without model_live is refused under "live".
    python tests/test_start_weights_live.py        (or pytest tests/test_start_weights_live.py)
"""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import gzip
import os
import sys
import tempfile

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_experiment import BaseExperiment  # noqa: E402


class _Stub:
    _load_pretrained_weights = BaseExperiment._load_pretrained_weights

    def __init__(self, start):
        self.cfg = OmegaConf.create({"seed": 42, "training": {"loss": "MSE"},
                                     "fine_tune": {"start_weights": start, "grow_width": False, "shrink_perturb": None}})
        self.model = torch.nn.Linear(3, 2)
        self.device = "cpu"


def _ckpt(tmp, live=True):
    ema = {k: torch.full_like(v, 1.0) for k, v in torch.nn.Linear(3, 2).state_dict().items()}
    lv = {k: torch.full_like(v, 2.0) for k, v in ema.items()}
    ck = {"model": ema, "model_is_ema": True}
    if live:
        ck["model_live"] = lv
    path = os.path.join(tmp, f"p{int(live)}.pt.gz")
    with gzip.open(path, "wb") as f:
        torch.save(ck, f)
    return path


def test_start_weights():
    with tempfile.TemporaryDirectory() as tmp:
        p = _ckpt(tmp)
        s = _Stub("live"); s._load_pretrained_weights(p)
        assert all(torch.all(v == 2.0) for v in s.model.state_dict().values()), "live weights not loaded"
        s = _Stub("scored"); s._load_pretrained_weights(p)
        assert all(torch.all(v == 1.0) for v in s.model.state_dict().values()), "EMA weights not loaded"
        try:
            _Stub("live")._load_pretrained_weights(_ckpt(tmp, live=False))
        except RuntimeError as e:
            assert "model_live" in str(e)
        else:
            raise AssertionError("a checkpoint without model_live was not refused")


if __name__ == "__main__":
    test_start_weights()
    print("ok")
