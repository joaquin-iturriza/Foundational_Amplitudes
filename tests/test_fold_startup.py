"""A search whose folded observations (extend_sweep --fold-points) already cover its random start-up suggests a
guided point, never its start-up indices again; a fresh search still starts with them (both surrogates)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep.dyhpo_sampler import DyHPOSampler

SPACE = [{"name": "training.lr", "type": "float_log", "low": 1e-4, "high": 1e-2},
         {"name": "training.ema_decay", "type": "float_uniform", "low": 0.9, "high": 0.999}]


def sampler(surrogate, tmp):
    return DyHPOSampler(SPACE, {"t_steps": [100]}, n_candidates=50, seed=42, output_path=str(tmp), n_startup=3,
                        surrogate=surrogate)


def check(surrogate, tmp):
    fresh = sampler(surrogate, tmp)
    first = int(fresh.algorithm.init_conf_indices[0])
    assert fresh.suggest()[0] == first
    s = sampler(surrogate, tmp)
    init = [int(i) for i in s.algorithm.init_conf_indices]
    for lr, v in ((2e-4, 0.5), (1e-3, 0.2), (5e-3, 0.4)):   # three outside points: the start-up is covered
        s.add_observed({"training.lr": lr, "training.ema_decay": 0.95}, 100, v)
    h = s.suggest()[0]
    assert s.algorithm.initial_random_index == len(init)
    assert h not in init, (surrogate, h, init)


def test_fold_skips_startup_gp(tmp_path):
    check("gp", tmp_path)


def test_fold_skips_startup_dyhpo(tmp_path):
    check("dyhpo", tmp_path)


if __name__ == "__main__":
    import tempfile
    for sg in ("gp", "dyhpo"):
        check(sg, tempfile.mkdtemp())
    print("ok")
