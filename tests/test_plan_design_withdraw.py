"""plan design-withdraw (the user rejected a recorded design in conversation) and plan review with a prompt passed by
file path. Scratch plan, registry and fixture transcripts; nothing live is touched."""
import datetime as dt, importlib, json, os, sqlite3, sys, tempfile
import pytest, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def world():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "plan")); os.makedirs(os.path.join(d, "agents"))
    db = os.path.join(d, "runs.db")
    c = sqlite3.connect(db)
    c.execute("create table decisions (id, project, kind, in_force, covers, said_at, quote, note)")
    c.execute("insert into decisions values ('D7','FA','prefer',1,'e1','2026-11-01T00:00:00+00:00','no, not that','')")
    c.execute("create table items (project, id, check_cmd, decision)")
    c.execute("create table runs (id, project, item, state, finished)")
    c.execute("create table events (id, ts, project, item, kind, key, msg, acked, ack_note, mute_until, tell)")
    c.commit()
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=3)).isoformat()
    design = dict(at=old, **{k: "x" for k in ("question", "prior", "measure", "check", "fixed", "control", "seeds",
                                               "outcome")})
    e = dict(id="e1", title="t", question="q", status="running", next="n", designs=[design])
    yaml.safe_dump({"entries": [e]}, open(os.path.join(d, "plan", "plan.yaml"), "w"))
    open(os.path.join(d, "plan", ".state.json"), "w").write("{}")
    os.environ.update(PLAN_TEST="1", PLAN_ROOT=d, PLAN_RUNS_DB=db, PLAN_AGENTS=os.path.join(d, "agents"),
                      PLAN_EVENTS_CMD="true")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import plan
    return d, importlib.reload(plan)


def entry(plan):
    return plan.entry(plan.load()[0], "e1")


class A:
    def __init__(self, **k):
        self.__dict__.update(k)


def reviews(plan, d, e):
    return plan.next_steps_keyed(*plan.load(), msgs=[])


def has_review(plan):
    return any(str(why).startswith("design to review") for _, why, _, _ in plan.next_steps_keyed(*plan.load(), msgs=[]))


def test_withdrawn_gives_no_review():
    d, plan = world()
    assert has_review(plan)                                   # unwithdrawn: the review is demanded
    plan.cmd_design_withdraw(A(id="e1", decision="D7", why="rejected in conversation"))
    x = plan.last_design(entry(plan))
    assert x["withdrawn_decision"] == "D7" and not plan.design_pending(entry(plan))
    assert not has_review(plan)
    assert "withdrawn" in plan.design_text(x)
    with pytest.raises(SystemExit):                           # nothing left to withdraw
        plan.cmd_design_withdraw(A(id="e1", decision="D7", why="again"))


def test_bad_decision_refused():
    d, plan = world()
    with pytest.raises(SystemExit) as r:
        plan.cmd_design_withdraw(A(id="e1", decision="D999", why="x"))
    assert "not a recorded decision" in str(r.value)
    assert plan.design_pending(entry(plan)) and has_review(plan)


def agent(d, aid, prompt):
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    open(os.path.join(d, "agents", "agent-%s.jsonl" % aid), "w").write(
        json.dumps({"type": "user", "timestamp": ts, "message": {"content": prompt}}) + "\n")


def test_review_prompt_by_file():
    d, plan = world()
    f = os.path.join(d, "review_prompt.txt")
    open(f, "w").write(plan.review_prompt(entry(plan)) + "\n")
    agent(d, "good1", "Read %s and follow it." % f)
    plan.cmd_review(A(id="e1", agent="good1"))
    assert plan.load()[1]["e1"]["reviews"][0]["agent"] == "good1"
    g = os.path.join(d, "other.txt")
    open(g, "w").write("something else entirely, Entry: e1\n")
    agent(d, "bad1", "Read %s and follow it." % g)
    with pytest.raises(SystemExit):
        plan.cmd_review(A(id="e1", agent="bad1"))
    agent(d, "inline1", plan.review_prompt(entry(plan)))      # the inline form still passes
    plan.cmd_review(A(id="e1", agent="inline1"))
