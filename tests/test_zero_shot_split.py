"""tools/zero_shot_parent.py --split: the role a probe read scores (rebuild_run.scored_split). The default stays val, so
every read taken so far reproduces; test scores the test loader, checks the test pool's physics and takes the test
events' target factors. CPU, no model built.
    .venv-cpu/bin/pytest tests/test_zero_shot_split.py"""
import os, subprocess, sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from rebuild_run import scored_split  # noqa: E402


def _exp():
    return SimpleNamespace(val_loader="VAL", test_loader="TEST",
                           _pool_paths={"val": {"p": "p_val.npy"}, "test": {"p": "p_test.npy"}},
                           _split_indices={"train": [0, 1], "val": [2, 3], "test": [4, 5]})


def test_val_and_test_roles():
    e = _exp()
    assert scored_split(e, "val", "p") == ("VAL", "p_val.npy", [2, 3])
    assert scored_split(e, "test", "p") == ("TEST", "p_test.npy", [4, 5])
    assert scored_split(e, "test", "missing")[1] is None          # no pool path: the physics check reports it
    try:
        scored_split(e, "train", "p")
    except AssertionError:
        pass
    else:
        raise AssertionError("the train role is no held-out read")


def test_cli_default_is_val():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "zero_shot_parent.py"), "--help"],
                         capture_output=True, text=True, env={**os.environ, "COLUMNS": "200"})
    assert out.returncode == 0, out.stderr
    assert "--split {val,test}" in out.stdout and "default, every read so far" in out.stdout


def test_reader_and_job_use_the_split():
    zs = open(os.path.join(ROOT, "tools", "zero_shot_parent.py")).read()
    assert "scored_split(exp, a.split" in zs and "exp.val_loader" not in zs and '_split_indices["val"]' not in zs
    job = open(os.path.join(ROOT, "scripts", "job_catalog_v3_short.sh")).read()
    assert '--split "$SPLIT"' in job and 'SPLIT=val;' in job and '[ "$A" = PP4 ]' in job


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
