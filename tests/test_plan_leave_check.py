"""Tonight's case (2026-11-09): agent_redesign's owner left steps 18 and 19 (waiting on the repo-reviewer) and 21 (an
after-period on or after 2026-11-17) with no `plan wait`, and the main session's stop was refused once per step.
  - plan leave-check lists every uncovered step in one refusal (18, 19 and 21 together);
  - entry_reason (what plan_check's RESUME/LAUNCH line says) names all ready steps at once;
  - a date wait (`plan wait 21 --on 2026-11-17`) takes step 21 off both until the date, and back after it;
  - plan_check.py on SubagentStop (D337): the owner is refused once naming 18, 19 and 21; with every step waited on
    and 21 on a date not yet come it passes, and is refused for 21 once its date has come; a non-owner agent passes.
Runs on a copy of the plan at e123833a^ (before steps 18 and 19 were dropped) (a temporary world, PLAN_TEST=1); the live plan is
never written.

    .venv-cpu/bin/python tests/test_plan_leave_check.py
"""
import datetime as dt, importlib, json, os, re, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="leave_check_")
PLAN = os.path.join(TMP, "plan", "plan.yaml")
COMMIT = "e123833a^"    # plan after `add-next` of step 21 and before steps 18 and 19 were dropped
src = subprocess.run(["git", "-C", ROOT, "show", "%s:plan/plan.yaml" % COMMIT], capture_output=True, text=True).stdout
os.makedirs(os.path.dirname(PLAN))
with open(PLAN, "w") as fh:
    fh.write(src)
WORLD = dict(PLAN_TEST="1", PLAN_ROOT=TMP, PLAN_SITE="true", PLAN_TRANSCRIPTS=os.path.join(TMP, "none"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import plan  # noqa: E402


def world():
    """plan (re)loaded on this world, the environment restored after: plan reads its PLAN_* at import, and a value
    left in os.environ (PLAN_FILE, before) pointed the next test file's plan at this world (2026-11-10:
    tests/test_plan_design_withdraw.py failed 'no entry e1' after this file); another file's reload of plan is undone."""
    global plan
    old = {k: v for k, v in os.environ.items() if k.startswith("PLAN_")}   # another file's PLAN_FILE left set too
    for k in old:
        del os.environ[k]
    os.environ.update(WORLD)
    try:
        plan = importlib.reload(plan)
    finally:
        for k in WORLD:
            os.environ.pop(k, None)
        os.environ.update(old)
    plan.commit_plan = lambda: None
    plan.item_checks = lambda ids: {}
    plan.site_sweeps = lambda *a, **k: {}


def steps(bad):
    return sorted(int(b.split(":")[0].split()[1]) for b in bad)


def setup_module():
    world()
    with plan.transaction() as (d, st):
        e = plan.entry(d, "agent_redesign")
        e["step_waits"] = {}                       # as the owner left it: nothing waited


def test_all_uncovered_steps_in_one_list():
    setup_module()
    ks = set(dict(plan.nexts(plan.entry(plan.load()[0], "agent_redesign"))))
    assert {18, 19, 21} <= ks, ks
    got = steps(plan.leave_problems("agent_redesign"))
    assert {18, 19, 21} <= set(got), got


def test_entry_reason_names_every_ready_step():
    setup_module()
    d, st = plan.load()
    e = plan.entry(d, "agent_redesign")
    r = plan.entry_reason(e, st, plan.messages(plan.oldest_open(d, st)), plan.now(), d, {}, {})
    assert r and r[0].startswith("ready steps ") and all(" %d" % k in r[0].split(" of ")[0].replace(",", " ")
                                                         for k in (18, 19, 21)), r


def test_date_wait():
    setup_module()
    with plan.transaction() as (d, st):
        e = plan.entry(d, "agent_redesign")
        e["step_waits"] = {"18": "the repo-reviewer's verdict", "19": "the repo-reviewer's verdict",
                           "21": "on or after 2026-11-17"}
    assert plan.wait_date("on or after 2026-11-17") == dt.datetime(2026, 11, 17, tzinfo=dt.timezone.utc)
    assert plan.wait_date("FA-ccin2p3-a77e20b8 2026-11-17") is None, "a wait on runs is not a date wait"
    assert plan.wait_date("2026-11-17") and plan.wait_date("the date 2026-11-17: a week of usage")
    # E4458 (2026-11-10): waits that only date something else are not date waits
    for w in ("a day of clean running of switch-overs A to D (applied 2026-10-30 about 12:30)",
              "Joaquin: syncClaudeAiSkills false (asked 2026-10-31 ~00:30)", "reviewer, expected 2026-11-12"):
        assert plan.wait_date(w) is None, w
    before = dt.datetime(2026, 11, 10, tzinfo=dt.timezone.utc)
    after = dt.datetime(2026, 11, 18, tzinfo=dt.timezone.utc)
    assert 21 not in steps(plan.leave_problems("agent_redesign", t=before))
    assert 21 in steps(plan.leave_problems("agent_redesign", t=after))
    d, st = plan.load()
    e = plan.entry(d, "agent_redesign")
    msgs = plan.messages(plan.oldest_open(d, st))
    r = plan.entry_reason(e, st, msgs, before, d, {}, {})
    assert not r or "21" not in r[0].split(" of ")[0].replace(",", " ").split(), r
    r = plan.entry_reason(e, st, msgs, after, d, {}, {})
    assert r and "step 21 waited until 2026-11-17" in r[0], r


OWNER = "a3433fbc56e65e68c"     # agent_redesign's recorded owner at e123833a^


def hook(aid, event="SubagentStop"):
    """.claude/hooks/plan_check.py through the dispatcher's input, as Claude Code sends a SubagentStop."""
    env = dict({k: v for k, v in os.environ.items() if not k.startswith("PLAN_")}, **WORLD, PLAN_EVENTS_CMD="true")
    inp = {"hook_event_name": event, "agent_id": aid, "agent_type": "general-purpose", "session_id": "s"}
    r = subprocess.run([sys.executable, os.path.join(ROOT, ".claude", "hooks", "plan_check.py")],
                       input=json.dumps(inp), capture_output=True, text=True, env=env, cwd=TMP, timeout=600)
    return r.returncode, r.stderr


def test_subagent_stop_owner_refused_once_with_every_step():
    setup_module()
    rc, err = hook(OWNER)
    lines = [l for l in err.splitlines() if "steps you leave uncovered" in l]
    assert rc == 2 and len(lines) == 1, err
    named = {int(x) for x in re.findall(r"step (\d+):", lines[0])}
    assert {18, 19, 21} <= named, err


def test_subagent_stop_owner_passes_with_waits_and_a_date_wait():
    setup_module()
    with plan.transaction() as (d, st):
        e = plan.entry(d, "agent_redesign")
        e["step_waits"] = {str(k): "x" for k, _ in plan.nexts(e)}
        e["step_waits"]["21"] = "on or after 2099-01-01"         # a date not yet come: quiet
    rc, err = hook(OWNER)
    assert rc == 0, err
    with plan.transaction() as (d, st):
        plan.entry(d, "agent_redesign")["step_waits"]["21"] = "on or after 2020-01-01"   # its date has come
    rc, err = hook(OWNER)
    named = {int(x) for x in re.findall(r"step (\d+):", err)}
    assert rc == 2 and named == {21}, err


def test_subagent_stop_non_owner_passes():
    setup_module()                                 # every step uncovered, but the agent owns no entry
    rc, err = hook("a" + "0" * 16)
    assert rc == 0, err



def test_subagent_stop_corrupt_plan_refuses_only_the_owner():
    setup_module()
    good = open(PLAN).read()
    try:
        with open(PLAN, "w") as fh:
            fh.write(good + "\n  : [unclosed\n")           # no longer parses
        rc, err = hook("a" + "0" * 16)
        assert rc == 0, err                              # a reviewer or worker cannot repair the plan: passes
        rc, err = hook(OWNER)
        assert rc == 2 and "leave-check failed" in err, err
    finally:
        with open(PLAN, "w") as fh:
            fh.write(good)


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); n += 1; print("ok", k, flush=True)
    print("%d passed" % n)
