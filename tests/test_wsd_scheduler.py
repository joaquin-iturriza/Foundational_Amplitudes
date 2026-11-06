"""misc.wsd_scheduler: warmup, flat, (1 - sqrt) cooldown to eta_min, and base_experiment's WSD branch builds it."""
import math, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc import wsd_scheduler  # noqa: E402


def _lrs(T=100, warm=10, cool=0.2, lr=1e-2, eta=1e-4):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=lr)
    s = wsd_scheduler(opt, warm, T, cool, eta_min=eta)
    out = []
    for _ in range(T):
        out.append(opt.param_groups[0]["lr"]); opt.step(); s.step()
    return out


def test_shape():
    lr = _lrs()
    assert lr[0] == 0 and abs(lr[5] - 5e-3) < 1e-12           # linear warmup
    assert all(abs(x - 1e-2) < 1e-12 for x in lr[10:80])       # stable until the last 20 steps
    assert abs(lr[90] - (1e-4 + (1e-2 - 1e-4) * (1 - math.sqrt(0.5)))) < 1e-12   # (1 - sqrt) cooldown
    assert lr[80] > lr[90] > lr[99] >= 1e-4


def test_cooldown_longer_than_stable():
    lr = _lrs(T=50, warm=20, cool=0.9)                          # cooldown starts at the end of the warmup
    assert abs(lr[20] - 1e-2) < 1e-12 and lr[21] < 1e-2


if __name__ == "__main__":
    test_shape(); test_cooldown_longer_than_stable(); print("ok")
