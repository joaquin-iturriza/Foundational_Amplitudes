"""CPU check of val_total, the sign-head search's objective (experiment._val_total_per_process, _val_total;
sweep/run_trial.py `objective:`): on the same events, per process, it is the loss the model trains on
(_aggregate_per_process_loss with the sign head), combined over processes by the validation aggregation (the training's
own aggregation does not move it); the near-zero weight (training only) does not move it; monitor processes are left
out; it is None without the sign head; run_trial observes the configured result key, fails a result without it, and
refuses a search over sign_loss_weight with this objective; the step-0 validation of a fine-tune is taken back out
of val_total too (base_experiment._START_CURVES), so val_total stays aligned with val_loss_no_reg's best index; a
detached trial's RESULT_JSON reports val_loss_no_reg as `val_loss` and the objective as `objective_value`, and
run_trial.result_fields reads both the new lines and the ones written before the fix.
    python tests/test_val_total.py        (or pytest tests/test_val_total.py)
"""
import os
import sys
import types

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sweep"))

from experiment import AmplitudeExperiment as E  # noqa: E402


def _stub(agg="mean", near_w=1.0, val_agg="geometric_mean"):
    names = ["p0", "p1", "p2", "p3"]
    s = types.SimpleNamespace()
    s.cfg = OmegaConf.create({"training": {"loss": "MSE", "loss_aggregation": agg, "sign_loss_weight": 1.0,
                                           "sign_near_weight": near_w, "sign_near_factor": 10.0,
                                           "val_aggregation": val_agg},
                              "data": {"dataset": names}})
    s.n_datasets = len(names)
    s._sign_head = True
    s._proc_signed = np.array([0, 1, 0, 1], dtype=np.float32)
    s.model = types.SimpleNamespace(training=False)
    s._per_event_loss = types.MethodType(E._per_event_loss, s)
    return s


def _batch(seed=0, n=400):
    g = torch.Generator().manual_seed(seed)
    pids = torch.arange(n) % 4
    y_pred, y = torch.randn(n, 1, generator=g), torch.randn(n, 1, generator=g)
    logit = torch.randn(n, 1, generator=g)
    target = (torch.rand(n, 1, generator=g) > 0.5).float()
    return y_pred, y, pids, logit, target


def _val_side(s, y_pred, y, pids, logit, target, monitors=()):
    per = E._per_event_loss(s, y_pred, y).double()
    sums = torch.zeros(4, dtype=torch.float64).index_add_(0, pids, per)
    counts = torch.zeros(4, dtype=torch.float64).index_add_(0, pids, torch.ones_like(per))
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit.reshape(-1), target.reshape(-1),
                                                               reduction="none").double()
    bces = torch.zeros(4, dtype=torch.float64).index_add_(0, pids, bce)
    s._val_total_proc = E._val_total_per_process(s, sums.numpy(), counts.numpy(), bces.numpy())
    agg = s.cfg.training.get("val_aggregation", "geometric_mean")
    comb = (lambda v: float(np.mean(v))) if agg == "mean" else None   # None: the geometric mean, _validate's default
    return E._val_total(s, monitors, comb)


def test_equals_training_loss():
    s = _stub(val_agg="mean")
    y_pred, y, pids, logit, target = _batch()
    train = float(E._aggregate_per_process_loss(s, y_pred, y, pids, "mean", sign=(logit, target)))
    val = _val_side(s, y_pred, y, pids, logit, target)
    assert abs(train - val) < 1e-6 * abs(train), (train, val)
    # the sign term is really in it: the magnitude-only mean is lower
    mse_only = float(torch.stack([((y_pred - y) ** 2)[pids == p].mean() for p in range(4)]).mean())
    assert val > mse_only + 0.1


def test_validation_aggregation_decides():
    s = _stub("mean")      # training aggregation mean, validation geometric mean
    y_pred, y, pids, logit, target = _batch(1)
    train_gm = float(E._aggregate_per_process_loss(s, y_pred, y, pids, "geometric_mean", sign=(logit, target)))
    v = _val_side(s, y_pred, y, pids, logit, target)
    assert abs(train_gm - v) < 1e-6 * train_gm
    assert _val_side(_stub("geometric_mean"), y_pred, y, pids, logit, target) == v


def test_near_weight_and_monitors():
    a, b = _stub(near_w=1.0, val_agg="mean"), _stub(near_w=30.0, val_agg="mean")
    batch = _batch(2)
    assert _val_side(a, *batch) == _val_side(b, *batch)
    full = _val_side(a, *batch)
    no_p1 = _val_side(a, *batch, monitors=("p1",))
    per = a._val_total_proc
    assert abs(no_p1 - np.mean([per["p0"], per["p2"], per["p3"]])) < 1e-12 and no_p1 != full
    a._sign_head = False
    assert E._val_total(a) is None


def test_run_trial_objective():
    import run_trial as rt
    res = {"val_loss": 0.01, "val_total": 0.05}
    assert rt._objective_value({}, res) == 0.01
    assert rt._objective_value({"objective": "val_total"}, res) == 0.05
    try:
        rt._objective_value({"objective": "val_total"}, {"val_loss": 0.01})
    except KeyError:
        pass
    else:
        raise AssertionError("a result without the configured objective must fail the trial")
    cfg = rt.load_config(os.path.join(ROOT, "sweep", "sweep_config_ccin2p3_sign_head_search.yaml"))
    assert cfg["objective"] == "val_total" and cfg["fidelity_schedule"]["t_steps"] == [4264]
    rt._check_objective(cfg)
    bad = dict(cfg, search_space=cfg["search_space"] + [{"name": "training.sign_loss_weight", "type": "float_log",
                                                         "low": 0.1, "high": 10}])
    try:
        rt._check_objective(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("val_total with sign_loss_weight searched must be refused")


def _start_stub(with_vt):
    """A fine-tune-like experiment with two validations done, whose _validate appends like experiment._validate
    (val_total rebound, not appended in place)."""
    from base_experiment import BaseExperiment as B
    s = types.SimpleNamespace(_START_CURVES=B._START_CURVES, _START_PER_PROCESS=B._START_PER_PROCESS,
                              train_sampler=None, val_loss=[3.0, 2.0], val_loss_no_reg=[3.0, 2.0], val_mse=[],
                              proc_val_losses={"p0": [3.0, 2.0]}, proc_val_losses_no_reg={"p0": [3.0, 2.0]})
    if with_vt:
        s.val_total = [5.0, 4.0]

    def _validate(step):
        s.val_loss.append(1.0)
        s.val_loss_no_reg.append(1.0)
        s.proc_val_losses["p0"].append(1.0)
        s.proc_val_losses_no_reg["p0"].append(1.0)
        s.val_total = getattr(s, "val_total", []) + [0.5]
    s._validate = _validate
    return s, B


def test_validate_start_val_total():
    for with_vt in (False, True):
        s, B = _start_stub(with_vt)
        assert B._validate_start(s) == 1.0
        assert s.val_loss_no_reg == [3.0, 2.0] and s.proc_val_losses["p0"] == [3.0, 2.0]
        if with_vt:
            assert s.val_total == [5.0, 4.0], s.val_total
        else:
            assert not hasattr(s, "val_total"), "a step-0 val_total must not survive as the first entry"
    # a fresh run: step-0 then the training validations; val_total read at val_loss_no_reg's best index is that step's
    s, B = _start_stub(False)
    s.val_loss, s.val_loss_no_reg, s.proc_val_losses["p0"], s.proc_val_losses_no_reg["p0"] = [], [], [], []
    B._validate_start(s)
    for nr, vt in ((2.0, 7.0), (1.0, 6.0), (1.5, 6.5)):
        s.val_loss_no_reg.append(nr)
        s.val_total = getattr(s, "val_total", []) + [vt]
    assert s.val_total[int(np.argmin(s.val_loss_no_reg))] == 6.0


def test_result_json_fields():
    import run_trial as rt
    # after the fix
    new = {"val_loss": 0.01, "objective": "val_total", "objective_value": 0.05, "observe_loss": 0.05}
    assert rt.result_fields(new) == (0.01, "val_total", 0.05)
    # a line written before the fix by a val_total search: val_loss carried the objective
    old = {"val_loss": 0.05, "objective": "val_total", "observe_loss": 0.05}
    assert rt.result_fields(old) == (None, "val_total", 0.05)
    old_default = {"val_loss": 0.01, "objective": "val_loss", "observe_loss": 0.01}
    assert rt.result_fields(old_default) == (0.01, "val_loss", 0.01)
    assert rt.result_fields({"val_loss": 0.01, "observe_loss": 0.01}) == (0.01, "val_loss", 0.01)


def test_detached_result_json():
    """run_trial.main --detached on a stubbed run.py: the printed RESULT_JSON names each number by what it is."""
    import contextlib
    import io
    import json
    import tempfile
    import run_trial as rt
    tmp = tempfile.mkdtemp(prefix="zz_val_total_")
    for objective, want_obj in (("val_total", 0.05), (None, 0.01)):
        cfg = {"sweep_name": "zz_sweep", "paths": {"project_dir": tmp, "sweep_dir": tmp},
               "fidelity_schedule": {"t_steps": [10]}}
        if objective:
            cfg["objective"] = objective
        seen = {}

        def build(cfg_, hp, run_dir, run_idx, result_path, *a, **k):
            seen["rp"] = result_path
            return ["true"]

        def run(cmd, cwd=None):
            with open(seen["rp"], "w") as f:
                json.dump({"val_loss": 0.01, "val_total": 0.05, "proc_val_losses": {"p0": 0.01}}, f)
            return types.SimpleNamespace(returncode=0)

        saved = (rt.load_config, rt.build_command, rt.subprocess.run, sys.argv)
        rt.load_config, rt.build_command, rt.subprocess.run = (lambda p: dict(cfg)), build, run
        sys.argv = ["run_trial.py", "--sweep-config", "x.yaml", "--trial-idx", "0", "--detached",
                    "--hp-idx", "3", "--t-steps", "10", "--hp", "training.lr=0.001"]
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                rt.main()
        finally:
            rt.load_config, rt.build_command, rt.subprocess.run, sys.argv = saved
        line = [l for l in out.getvalue().splitlines() if l.startswith("RESULT_JSON ")][-1]
        res = json.loads(line[len("RESULT_JSON "):])
        assert res["val_loss"] == 0.01, res                       # always val_loss_no_reg
        assert res["objective"] == (objective or "val_loss") and res["objective_value"] == want_obj, res
        assert res["observe_loss"] == want_obj, res
        assert rt.result_fields(res) == (0.01, objective or "val_loss", want_obj)
    import shutil
    shutil.rmtree(tmp)                                            # the test's own temporary dir only


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print(k, "ok")
