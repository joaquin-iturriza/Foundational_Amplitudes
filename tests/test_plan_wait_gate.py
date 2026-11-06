"""`plan wait` naming the user needs an open `plan ask` holding that step (2026-11-05: catalog_coverage and warm_start
recorded eight waits on Joaquin's answers that were never asked; nothing woke the main session for 19 h)."""
import argparse, os, sys, yaml
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(tmp):
    os.environ.update(PLAN_TEST="1", PLAN_ROOT=str(tmp), PLAN_OWNER_MODE="1")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import importlib, plan
    return importlib.reload(plan)


def put(tmp, **kw):
    e = {"id": "ent", "status": "running", "next": {1: "A", 2: "B", 3: "C"}}
    e.update(kw)
    os.makedirs(os.path.join(str(tmp), "plan"), exist_ok=True)
    with open(os.path.join(str(tmp), "plan", "plan.yaml"), "w") as f:
        yaml.safe_dump({"entries": [e], "migrated_at": "2026-10-01T00:00:00+00:00"}, f)


def wait(plan, n, on):
    plan.cmd_wait(argparse.Namespace(id="ent", n=n, on=on))


ASK = dict(ask="which way for the cache?", waiting_since="2026-11-05T10:00:00+00:00", blocks_steps=[2])


def test_run_id_passes(tmp_path):
    plan = load(tmp_path); put(tmp_path)
    wait(plan, 1, "FA-ccin2p3-12cb7292 FA-ccin2p3-37dd5af5")


def test_user_without_ask_refused(tmp_path):
    plan = load(tmp_path); put(tmp_path)
    with pytest.raises(SystemExit) as x:
        wait(plan, 1, "Joaquin's answer on the memmap cache")
    assert "no open `plan ask`" in str(x.value)


def test_user_with_ask_on_other_step_refused(tmp_path):
    plan = load(tmp_path); put(tmp_path, **ASK)
    for on in ("Joaquin's call on X", "the user's go", "his answer"):
        with pytest.raises(SystemExit):
            wait(plan, 1, on)


def test_user_with_matching_open_ask_passes(tmp_path):
    plan = load(tmp_path); put(tmp_path, **ASK)
    wait(plan, 2, "Joaquin's answer on the memmap cache")


def test_answered_ask_refused(tmp_path):
    plan = load(tmp_path); put(tmp_path, **{**ASK, "ask": None})
    with pytest.raises(SystemExit):
        wait(plan, 2, "Joaquin's call")
