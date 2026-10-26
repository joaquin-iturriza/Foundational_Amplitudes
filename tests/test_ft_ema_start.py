"""A fine-tune's start (base_experiment.BaseExperiment._restart_ema, _validate_start).

init_model builds the EMA on the fresh initialisation and a fine-tune loads its parent's weights afterwards; torch_ema's
shadow is a copy taken when it is built, so the validation (scored on the EMA average) blended the random weights in
until _restart_ema reset it at the start of train(). _validate_start scores the weights before any update and takes
its entry back out of every curve, so the step-0 value is never a checkpoint candidate. Needs the site env (torch_ema):

    site run ccin2p3 FA --item muon_test -- python tests/test_ft_ema_start.py
"""
import os
import sys
import types

import torch
from torch_ema import ExponentialMovingAverage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_experiment import BaseExperiment  # noqa: E402


def _stub():
    s = types.SimpleNamespace()
    for k in ("_START_CURVES", "_START_PER_PROCESS"):
        setattr(s, k, getattr(BaseExperiment, k))
    return s


def test_restart_ema_scores_the_loaded_weights():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.99)    # init_model: the EMA on the fresh initialisation
    parent = {k: torch.randn_like(v) for k, v in model.state_dict().items()}
    model.load_state_dict(parent)                                     # the fine-tune loads the parent afterwards

    def scored():
        with ema.average_parameters():
            return {k: v.clone() for k, v in model.state_dict().items()}

    ema.update()                                                      # one update with the live weights unchanged
    blend = scored()
    assert not all(torch.allclose(blend[k], parent[k]) for k in parent), "the bug: the shadow kept the random init"

    s = _stub()                                                       # the live weights are still the parent's
    s.ema, s.model, s.device = ema, model, torch.device("cpu")
    s.cfg = types.SimpleNamespace(training=types.SimpleNamespace(ema_decay=0.99))
    BaseExperiment._restart_ema(s)
    ema = s.ema
    for _ in range(3):
        ema.update()
    after = scored()
    assert all(torch.allclose(after[k], parent[k]) for k in parent), "after the restart the EMA is the parent's weights"


def test_validate_start_leaves_no_trace_in_the_curves():
    s = _stub()
    s.train_sampler = None
    s.val_loss, s.val_loss_no_reg, s.val_mse, s.val_loss_no_reg_all = [], [], [], []
    s.proc_val_losses, s.proc_val_losses_no_reg = {}, {}

    def _validate(step):
        assert step == -1
        s.val_loss.append(0.5)
        s.val_loss_no_reg.append(0.25)
        s.val_loss_no_reg_all = s.val_loss_no_reg_all + [0.3]          # the multi-process path reassigns this one
        s.train_eval_loss_no_reg = [0.1]                               # created on first use
        s.proc_val_losses.setdefault("p", []).append(0.5)
        s.proc_val_losses_no_reg.setdefault("p", []).append(0.25)
        return 0.5

    s._validate = _validate
    v0 = BaseExperiment._validate_start(s)
    assert v0 == 0.25
    assert s.val_loss == [] and s.val_loss_no_reg == [] and s.val_mse == [] and s.val_loss_no_reg_all == []
    assert not hasattr(s, "train_eval_loss_no_reg")
    assert s.proc_val_losses == {} and s.proc_val_losses_no_reg == {}

    s.train_sampler = object()                                         # the balanced sampler: skipped, nothing touched
    assert BaseExperiment._validate_start(s) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
