"""drive.py --retry-failed: a failed trial whose run id or error matches is requeued with its penalty removed;
others keep theirs; lost ones only under --retry-lost."""
import sys, types, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep import drive


def _sweep():
    alg = types.SimpleNamespace(observations={1: {"c": 9.0}, 2: {"c": 9.0}, 3: {"c": 9.0}}, diverged_configs={1, 2, 3})
    smp = types.SimpleNamespace(algorithm=alg, _to_combo=lambda t: "c", _eval_order=[(1, "c"), (2, "c"), (3, "c")],
                                _val_loss_history={1: {"c": 9.0}, 2: {"c": 9.0}, 3: {"c": 9.0}})
    trials = {"0": dict(idx=0, hp_idx=1, t_steps=10, status="failed", rid="FA-x-aaa", error="run.py exited 1"),
              "1": dict(idx=1, hp_idx=2, t_steps=10, status="failed", rid="FA-x-bbb", error="diverged"),
              "2": dict(idx=2, hp_idx=3, t_steps=10, status="lost", rid="FA-x-ccc", error="")}
    s = types.SimpleNamespace(sampler=smp, st={"trials": trials}, name="t", save=lambda: None)
    s.trials = lambda *st: [t for t in trials.values() if not st or t["status"] in st]
    return s


def test_retry_failed_matches_only():
    s = _sweep()
    drive.Sweep.requeue_lost(s, False, "aaa")
    st = {t["idx"]: t["status"] for t in s.st["trials"].values()}
    assert st == {0: "retry", 1: "failed", 2: "lost"}
    assert 1 not in s.sampler.algorithm.observations and 2 in s.sampler.algorithm.observations
    assert s.sampler._eval_order == [(2, "c"), (3, "c")]


if __name__ == "__main__":
    test_retry_failed_matches_only(); print("ok")
