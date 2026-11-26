"""plan dead-end ends one attempt, never the work asked for: it needs --next or --asked with an open ask."""
import argparse, os, sys, yaml
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(tmp):
    os.environ.update(PLAN_TEST="1", PLAN_ROOT=str(tmp), PLAN_OWNER_MODE="1")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import importlib, plan
    return importlib.reload(plan)


def put(tmp, **kw):
    e = {"id": "ent", "status": "running", "next": {1: "A"}}
    e.update(kw)
    os.makedirs(os.path.join(str(tmp), "plan"), exist_ok=True)
    with open(os.path.join(str(tmp), "plan", "plan.yaml"), "w") as f:
        yaml.safe_dump({"entries": [e], "migrated_at": "2026-10-01T00:00:00+00:00"}, f)


def de(plan, next=None, asked=False, seq=None):
    plan.cmd_dead_end(argparse.Namespace(id="ent", tried="t", why="w", next=next, asked=asked, seq=seq))


def ent(tmp):
    return yaml.safe_load(open(os.path.join(str(tmp), "plan", "plan.yaml")))["entries"][0]


def test_bare_refused(tmp_path):
    plan = load(tmp_path); put(tmp_path)
    with pytest.raises(SystemExit) as x:
        de(plan)
    assert "never the work asked for" in str(x.value) and not ent(tmp_path).get("dead_ends")


def test_next_adds_step(tmp_path):
    plan = load(tmp_path); put(tmp_path)
    seq = [("feeds", "whether the per-event request works at all"), ("if_", "it works: report it"),
           ("if_", "it fails: try the other model")]
    de(plan, next="try the other model", seq=seq)
    e = ent(tmp_path)
    assert len(e["dead_ends"]) == 1 and "try the other model" in str(e["next"])


def test_asked_needs_open_ask(tmp_path):
    plan = load(tmp_path); put(tmp_path)
    with pytest.raises(SystemExit):
        de(plan, asked=True)
    put(tmp_path, asks=[{"ask": "which way?", "at": "2026-11-05T10:00:00+00:00"}])
    de(plan, asked=True)
    assert len(ent(tmp_path)["dead_ends"]) == 1
