"""CPU check of the sweep's zero-shot objective (sweep/run_trial.py zero_shot_score, _zero_shot_objective with
tools/zero_shot_parent.py replaced by a stub, _check_objective): the score is the geometric mean over classes of the
class geometric mean of the probes' MSE; a probe with an error, a non-finite MSE or a failed physics check fails the
trial; a missing probe fails it; the objective and its block go together.
    python tests/test_zero_shot_objective.py        (or pytest tests/test_zero_shot_objective.py)
"""
import json
import math
import os
import sys
import tempfile
import types

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sweep"))


def _line(p, x, st="ok", **kw):
    return "ZERO_SHOT " + json.dumps(dict(probe=p, mse_logm2_real=x, physics={"status": st}, **kw))


def test_score_by_class():
    import run_trial as rt
    lines = [_line("a1", 1.0), _line("a2", 4.0), _line("b", 0.5), "ZERO_SHOT_HIST {}", "other"]
    s, mse = rt.zero_shot_score(lines, {"a1": "A", "a2": "A", "b": "B"})
    # class A: gmean(1, 4) = 2; class B: 0.5; gmean(2, 0.5) = 1
    assert abs(s - 1.0) < 1e-12, s
    assert mse == {"a1": 1.0, "a2": 4.0, "b": 0.5}


def test_bad_probe_fails():
    import run_trial as rt
    for bad in (_line("b", float("nan")), _line("b", 1.0, st="mismatch"), _line("b", 1.0, error="boom"),
                _line("b", None)):
        try:
            rt.zero_shot_score([_line("a", 1.0), bad], {})
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    try:
        rt.zero_shot_score(["nothing"], {})
    except ValueError:
        return
    raise AssertionError("accepted an empty read")


def test_objective_runs_every_probe():
    import run_trial as rt
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "recipes")); os.makedirs(os.path.join(d, "run"))
    os.makedirs(os.path.join(d, "analysis", "catalog_v4"))
    json.dump({"classes": {"v4_x1": ["v4_x1", "v4_x2"]}}, open(os.path.join(d, "analysis", "catalog_v4",
                                                                            "census_classes.json"), "w"))
    yaml.safe_dump({"sampling": "mixture", "processes": [{"name": n, "n_test": 10} for n in ("hd_x1", "hd_x2", "hd_y")]},
                   open(os.path.join(d, "recipes", "r.yaml"), "w"))
    seen = {}

    def fake(cmd, cwd=None, capture_output=None, text=None, drop=None):
        files = [c for c in cmd if c.endswith(".yaml")]
        seen["cmd"] = cmd
        seen["tops"] = [yaml.safe_load(open(f)) for f in files]
        names = [y["processes"][0]["name"] for y in seen["tops"]]
        mse = {"hd_x1": 1.0, "hd_x2": 4.0, "hd_y": 0.5}
        out = "\n".join(_line(n, mse[n]) for n in names if n != fake.drop)
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    fake.drop = None
    rt.subprocess.run = fake
    zs = rt._zero_shot_objective({"recipe": "recipes/r.yaml"}, os.path.join(d, "run"), d)
    assert abs(zs["score"] - 1.0) < 1e-12 and zs["n_probes"] == 3 and zs["split"] == "test", zs
    assert all(t["sampling"] == "mixture" and len(t["processes"]) == 1 for t in seen["tops"])
    assert "--signed-magnitude" in seen["cmd"] and seen["cmd"][seen["cmd"].index("--split") + 1] == "test"
    assert json.load(open(os.path.join(d, "run", "zero_shot_objective.json")))["mse"]["hd_y"] == 0.5
    fake.drop = "hd_y"
    try:
        rt._zero_shot_objective({"recipe": "recipes/r.yaml"}, os.path.join(d, "run"), d)
    except ValueError:
        return
    raise AssertionError("accepted a read missing a probe")


def test_check_pairs():
    import run_trial as rt
    rt._check_objective({"objective": "zero_shot_score", "zero_shot_objective": {"recipe": "r"}})
    rt._check_objective({})
    for cfg in ({"objective": "zero_shot_score"}, {"zero_shot_objective": {"recipe": "r"}},
                {"objective": "zero_shot_score", "zero_shot_objective": {"recipe": "r"}, "cells": [{}]}):
        try:
            rt._check_objective(cfg)
        except ValueError:
            continue
        raise AssertionError(f"accepted {cfg}")


def test_real_classes():
    import run_trial as rt
    cls = rt._probe_classes(ROOT)
    c = json.load(open(os.path.join(ROOT, "analysis", "catalog_v4", "census_classes.json")))["classes"]
    rep, members = next((r, m) for r, m in c.items() if len(m) > 1)
    other = next(x for x in members if x != rep)
    assert cls("hd_" + other[len("v4_"):]) == rep


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
