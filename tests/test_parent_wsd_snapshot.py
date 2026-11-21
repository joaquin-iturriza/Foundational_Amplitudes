#!/usr/bin/env python
"""job_datasaving_parent.sh adds the WSD pre-cooldown step to training.save_model_at_steps: misc.wsd_cooldown_start is
the last step before wsd_scheduler's lr leaves 1, and the script wires it only when training.scheduler=WSD.
Run: python tests/test_parent_wsd_snapshot.py"""
import os, sys, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from misc import wsd_cooldown_start, wsd_scheduler


def lr_after(T, frac, n):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    s = wsd_scheduler(opt, 0, T, frac)
    for _ in range(n):
        opt.step(); s.step()
    return opt.param_groups[0]["lr"]


def main():
    for T, frac in [(64000, 0.125), (4307, 0.125), (1000, 0.3)]:
        n = wsd_cooldown_start(T, frac)
        assert n == T - round(frac * T)
        assert lr_after(T, frac, n) == 1.0 and lr_after(T, frac, n + 1) < 1.0, (T, frac)
    assert wsd_cooldown_start(64000, 0.125) == 56000
    sh = open(os.path.join(ROOT, "scripts", "job_datasaving_parent.sh")).read()
    assert 'training.scheduler=WSD' in sh and '"++training.save_model_at_steps=[$SNAP]"' in sh
    print("ok")


def test_parent_wsd_snapshot():
    main()


if __name__ == "__main__":
    main()
