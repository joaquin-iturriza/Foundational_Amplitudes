#!/usr/bin/env python
"""training.save_model_at_steps (plan warm_start, D289): BaseExperiment._save_at_step writes
models/model_run<idx>_step<N>.pt exactly at the listed steps (1-based, after the step), nothing otherwise, nothing when
the list is empty or missing; the file is _save_model's format, so with EMA on its "model" is the EMA average at that
step, and _load_pretrained_weights (the warm start's loader) starts from those EMA weights.
CPU, tiny model and a real torch_ema (the test_ema_checkpoint.py fixtures). Run: python tests/test_save_at_steps.py"""
import _tmpdir  # noqa: F401  (temp files under one dir removed at exit)
import os, sys, tempfile, types
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_ema_checkpoint import make_exp, BaseExperiment   # stubs the heavy imports, builds a tiny EMA experiment


def train_steps(e, n, steps):
    e.cfg.training = types.SimpleNamespace(save_model_at_steps=steps)
    e.is_main_process = lambda: True
    snap = {}
    for step in range(n):
        loss = e.model(torch.randn(16, 3)).pow(2).mean()
        e.optimizer.zero_grad(); loss.backward(); e.optimizer.step(); e.ema.update()
        e._save_at_step(step)
        with e.ema.average_parameters():
            snap[step + 1] = {k: v.detach().clone() for k, v in e.model.state_dict().items()}
    return snap


def main():
    with tempfile.TemporaryDirectory() as d:
        e = make_exp(d)
        snap = train_steps(e, 7, [3, 6])
        files = sorted(os.listdir(os.path.join(d, "models")))
        assert files == ["model_run0_step3.pt", "model_run0_step6.pt"], files
        for n in (3, 6):
            ck = torch.load(os.path.join(d, "models", f"model_run0_step{n}.pt"))
            assert ck["step"] == n and ck.get("model_is_ema"), (n, ck.keys())
            for k, v in ck["model"].items():
                assert torch.equal(v, snap[n][k]), (n, k)        # the EMA weights at that step
            assert any(not torch.equal(ck["model"][k], ck["model_live"][k]) for k in ck["model"])
        # the warm start's loader starts a fresh model from that file's EMA weights
        f = make_exp(os.path.join(d, "child"), seed=1)
        f._load_pretrained_weights(os.path.join(d, "models", "model_run0_step3.pt"))
        for k, v in f.model.state_dict().items():
            assert torch.equal(v, snap[3][k]), k
    for steps in ([], None):
        with tempfile.TemporaryDirectory() as d:
            e = make_exp(d)
            train_steps(e, 4, steps)
            assert os.listdir(os.path.join(d, "models")) == [], steps
    print("test_save_at_steps: ok")


if __name__ == "__main__":
    main()
