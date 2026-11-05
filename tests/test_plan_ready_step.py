"""Per-step readiness (2026-11-05, catalog_coverage sat with unstarted steps while one run was live and its owner
stopped): a step with no wait, no hold by an open ask and nothing of its own in flight is READY whatever else the
entry has live; with its owner not working, plan next lists a resume and the Stop gate refuses."""
import datetime as dt, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOKS = os.path.join(ROOT, ".claude", "hooks")


def load(tmp):
    os.environ.update(PLAN_TEST="1", PLAN_ROOT=str(tmp), PLAN_OWNER_MODE="1")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    sys.path.insert(0, HOOKS)
    import importlib, plan
    return importlib.reload(plan)


def world(plan, monkeypatch, owner_state, **kw):
    t = plan.now()
    e = {"id": "ent", "status": "running", "next": {1: "step A text", 2: "step B text"},
         "owner": {"agent": "aowner"}}
    e.update(kw)
    d = {"entries": [e], "migrated_at": (t - dt.timedelta(days=6)).isoformat()}
    monkeypatch.setattr(plan, "runs_of", lambda e_, st: [("FA-x-1", "RUNNING", None)])   # live, on step 1
    monkeypatch.setattr(plan, "item_checks", lambda eids: {i: 0 for i in eids})
    monkeypatch.setattr(plan, "site_sweeps", lambda *a, **k: {})
    monkeypatch.setattr(plan, "owner_state", lambda e_, t_=None: owner_state)
    monkeypatch.setattr(plan, "subagent_activity", lambda a: (t - dt.timedelta(hours=1), True))   # wrote after the date
    monkeypatch.setattr(plan, "agent_context", lambda a: {"tokens": 1000})
    monkeypatch.setattr(plan, "design_pending", lambda e_: False)
    return d, {}, t


def acts(plan, d, st, t):
    return plan.owner_actions(d, st, t=t, msgs=[], events={})


def test_live_run_and_ready_step_owner_stopped(tmp_path, monkeypatch):
    plan = load(tmp_path)
    d, st, t = world(plan, monkeypatch, "stopped")
    a = acts(plan, d, st, t)
    assert len(a) == 1 and a[0][1] == "resume" and "ready step 2 of ent has no one working on it" in a[0][3], a
    import plan_check
    out = plan_check.owner_problems(d, st, [])
    assert any("ready step 2 of ent has no one working on it" in x for x in out), out


def test_owner_working_not_flagged(tmp_path, monkeypatch):
    plan = load(tmp_path)
    d, st, t = world(plan, monkeypatch, "working")
    assert acts(plan, d, st, t) == []


def test_all_steps_waiting_or_blocked_not_flagged(tmp_path, monkeypatch):
    plan = load(tmp_path)
    d, st, t = world(plan, monkeypatch, "stopped", step_waits={1: "wait on FA-x-1"}, ask="q?",
                     waiting_since=plan.now().isoformat(), blocks_steps=[2])
    assert acts(plan, d, st, t) == []


def test_waiting_user_unchanged(tmp_path, monkeypatch):
    plan = load(tmp_path)
    d, st, t = world(plan, monkeypatch, "stopped", status="waiting_user", ask="q?",
                     waiting_since=plan.now().isoformat())
    assert acts(plan, d, st, t) == []
    assert [x for x in plan.entry_status(d, st, {}) if x[1] == "STALLED"] == []
