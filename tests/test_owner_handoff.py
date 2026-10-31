#!/usr/bin/env python3
"""Tests of .claude/hooks/owner_handoff.py (agent_redesign next 10): the PostToolUse note that tells an owner agent at
CONTEXT_CLOSE or more to write its hand-off and end. Synthetic transcripts and a synthetic plan only.

    .venv-cpu/bin/python -m pytest tests/test_owner_handoff.py        (or: python3 tests/test_owner_handoff.py)
    python3 tests/test_owner_handoff.py --replay <agent transcript>   (read in place, nothing copied: the first tool
        call at which the hook would have fired, against the transcript's compaction boundaries)

OWNER_HANDOFF_HOOK points the tests at a copy of the hook (it is built and tested in a copy before install)."""
import json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK = os.environ.get("OWNER_HANDOFF_HOOK") or os.path.join(ROOT, ".claude", "hooks", "owner_handoff.py")
OWNER, OTHER, CLOSED = "a0000000000000001", "a0000000000000002", "a0000000000000003"


def assistant(tokens, tool=True):
    c = [{"type": "tool_use", "id": "t", "name": "Bash", "input": {"command": "true"}}] if tool else \
        [{"type": "text", "text": "x"}]
    return {"type": "assistant", "message": {"role": "assistant", "content": c,
            "usage": {"input_tokens": 10, "cache_read_input_tokens": tokens - 1010,
                      "cache_creation_input_tokens": 1000, "output_tokens": 50}}}


def world(tmp, sizes):
    """A plan with entry e_own owned by OWNER (working) and e_done owned by CLOSED (hand-off written), and one
    transcript per agent whose last call is at sizes[agent] tokens, laid out as Claude Code lays out a session."""
    plan = {"entries": [
        {"id": "e_own", "status": "running", "owner": {"agent": OWNER, "state": "working"}},
        {"id": "e_done", "status": "running", "owner": {"agent": CLOSED, "state": "closed"}}]}
    import yaml
    with open(os.path.join(tmp, "plan.yaml"), "w") as f:
        yaml.safe_dump(plan, f)
    sess = os.path.join(tmp, "proj", "sess1", "subagents")
    os.makedirs(sess)
    main_t = os.path.join(tmp, "proj", "sess1.jsonl")
    with open(main_t, "w") as f:
        f.write(json.dumps(assistant(sizes.get("main", 10000))) + "\n")
    for a, n in sizes.items():
        if a == "main":
            continue
        with open(os.path.join(sess, "agent-%s.jsonl" % a), "w") as f:
            f.write(json.dumps({"type": "user", "message": {"role": "user", "content": "brief"}}) + "\n")
            f.write(json.dumps(assistant(n // 2)) + "\n")
            f.write(json.dumps({"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t", "content": "usage"}]}}) + "\n")
            f.write(json.dumps(assistant(n)) + "\n")
    return main_t


def run(tmp, inp, raw=None):
    env = dict(os.environ, PLAN_TEST="1", PLAN_ROOT=tmp, PLAN_FILE=os.path.join(tmp, "plan.yaml"),
               PLAN_AGENTS=os.path.join(tmp, "proj", "sess1", "subagents"), CLAUDE_PROJECT_DIR=ROOT)
    r = subprocess.run([sys.executable, HOOK], input=raw if raw is not None else json.dumps(inp),
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"] if r.stdout.strip() else None


def call(main_t, agent=None, **k):
    d = {"hook_event_name": "PostToolUse", "session_id": "sess1", "transcript_path": main_t, "tool_name": "Bash",
         "tool_input": {"command": "true"}, "tool_response": {}}
    if agent:
        d.update(agent_id=agent, agent_type="compact-worker")
    d.update(k)
    return d


def check(name, sizes, agent, want, **k):
    with tempfile.TemporaryDirectory() as tmp:
        main_t = world(tmp, sizes)
        got = run(tmp, call(main_t, agent, **k))
    ok = (got is None) if want is None else (got is not None and all(w in got for w in want))
    assert ok, "%s: got %r" % (name, got)


def test_main_session_never():
    check("main session at 190k", {"main": 190000, OWNER: 190000}, None, None)


def test_owner_below():
    check("owner at 120k", {OWNER: 120000}, OWNER, None)


def test_owner_just_below():
    check("owner at 169999", {OWNER: 169999}, OWNER, None)


def test_owner_at_line():
    check("owner at 170000", {OWNER: 170000}, OWNER, ["170000", "plan.py handoff e_own", "end"])


def test_owner_past_window():
    check("owner at 205k (past the window, compaction pending)", {OWNER: 205000}, OWNER, ["205000", "e_own"])


def test_non_owner_agent():
    check("reviewer at 190k owns nothing", {OTHER: 190000}, OTHER, None)


def test_closed_owner():
    check("owner whose hand-off is written", {CLOSED: 190000}, CLOSED, None)


def test_agent_transcript_path_field():
    with tempfile.TemporaryDirectory() as tmp:
        main_t = world(tmp, {OWNER: 100000})
        alt = os.path.join(tmp, "alt.jsonl")
        with open(alt, "w") as f:
            f.write(json.dumps(assistant(180000)) + "\n")
        got = run(tmp, call(main_t, OWNER, agent_transcript_path=alt))
    assert got and "180000" in got, got


def test_missing_transcript():
    with tempfile.TemporaryDirectory() as tmp:
        main_t = world(tmp, {})
        assert run(tmp, call(main_t, OWNER)) is None


def test_garbage_input():
    with tempfile.TemporaryDirectory() as tmp:
        world(tmp, {})
        assert run(tmp, None, raw="not json") is None
        assert run(tmp, None, raw="[1, 2]") is None
        assert run(tmp, None, raw="") is None


def test_last_record_not_assistant():
    """The tail ends in a tool result (the PostToolUse moment): the context is the last assistant call's."""
    check("owner 175k, tool result last", {OWNER: 175000}, OWNER, ["175000"])


def replay(path):
    """Walk an agent transcript in place: each assistant call with a tool_use is a PostToolUse moment; report the
    first at which the hook's threshold (CONTEXT_CLOSE) is met, and the compaction boundaries."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    sys.path.insert(0, os.path.dirname(HOOK))
    import plan, owner_handoff as oh
    lines, first, calls, comps, peak = [], None, 0, [], 0
    for raw in open(path, "rb"):
        lines.append(raw.rstrip(b"\n"))
        try:
            m = json.loads(raw)
        except ValueError:
            continue
        if m.get("type") == "system" and m.get("subtype") == "compact_boundary":
            comps.append((m.get("timestamp"), calls))
        if m.get("type") != "assistant":
            continue
        if not any(c.get("type") == "tool_use" for c in (m.get("message") or {}).get("content") or []
                   if isinstance(c, dict)):
            continue
        calls += 1
        n = oh.context_of(lines[-50:])
        peak = max(peak, n)
        if first is None and n >= plan.CONTEXT_CLOSE:
            first = (calls, n, m.get("timestamp"))
    print(json.dumps({"transcript": path, "tool_calls": calls, "peak_context": peak, "close_line": plan.CONTEXT_CLOSE,
                      "first_fire": first and {"call": first[0], "context": first[1], "at": first[2]},
                      "compactions": [{"at": t, "after_call": c} for t, c in comps]}, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--replay"]:
        replay(sys.argv[2])
        sys.exit(0)
    fails = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            try:
                f()
                print("PASS", k)
            except AssertionError as e:
                fails += 1
                print("FAIL", k, e)
    print("%d failed" % fails)
    sys.exit(1 if fails else 0)
