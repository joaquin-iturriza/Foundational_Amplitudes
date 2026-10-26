"""CPU check (run.py replaced by a stub) of a cells trial (sweep/run_trial.py _run_cells, the Muon fine-tune search
muon_hp_ft): each cell is one run.py with its own steps (iterations and the scheduler's increment) and its own params,
the objective is the geometric mean of the cells' val_loss_no_reg, and a failed cell fails the trial.
    python tests/test_run_trial_cells.py        (or pytest tests/test_run_trial_cells.py)
"""
import json
import math
import os
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sweep"))


def _arg(cmd, key):
    hit = [c.split("=", 1)[1] for c in cmd if c.split("=", 1)[0].lstrip("+") == key]
    assert len(hit) == 1, (key, hit)
    return hit[0]


def _run(fail_at=None):
    import run_trial as rt
    cfg = rt.load_config(os.path.join(ROOT, "sweep", "sweep_config_ccin2p3_muon_hp_ft.yaml"))
    cmds = []

    def fake(cmd, cwd=None):
        cmds.append(cmd)
        if fail_at is not None and len(cmds) == fail_at:
            return types.SimpleNamespace(returncode=1)
        json.dump({"val_loss": int(_arg(cmd, "training.iterations")) / 1000}, open(_arg(cmd, "training.result_path"), "w"))
        return types.SimpleNamespace(returncode=0)

    rt.subprocess.run = fake
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "res"))
    out = rt._run_cells(cfg, {"fine_tune.lr_scale": 2.0, "fine_tune.layer_decay": 0.9}, os.path.join(d, "run"),
                        os.path.join(d, "res"), 3, cfg["fidelity_schedule"]["t_steps"][0], d)
    return cfg, cmds, out


def test_cells():
    cfg, cmds, (obj, per, summary) = _run()
    assert len(cmds) == len(cfg["cells"]) == 16
    for c, cmd in zip(cfg["cells"], cmds):
        assert _arg(cmd, "training.iterations") == _arg(cmd, "training.increment_steps") == str(c["steps"])
        assert _arg(cmd, "data.train_subsample") == str(c["params"]["data.train_subsample"])
        assert _arg(cmd, "fine_tune.lr_scale").startswith("2.0") and _arg(cmd, "training.optimizer") == "MuonAdamW"
        assert _arg(cmd, "fine_tune.target_stats") == "parent_shared"
        assert _arg(cmd, "run_dir").endswith(c["name"])
    want = math.exp(sum(math.log(c["steps"] / 1000) for c in cfg["cells"]) / 16)
    assert abs(obj - want) < 1e-12 and json.load(open(summary))["val_loss"] == obj


def test_failed_cell():
    try:
        _run(fail_at=5)
    except RuntimeError as e:
        assert "run.py failed" in str(e)
    else:
        raise AssertionError("a failed cell must fail the trial")


if __name__ == "__main__":
    test_cells()
    test_failed_cell()
    print("ok")
