#!/usr/bin/env python3
"""Tests of context_guard.compacted_agent and the SessionStart branches of context_guard.py and context_push.py
(agent_redesign next 9): a subagent's compaction fires SessionStart:compact with the main session's ids and no
agent_id; the agent gets a short note of its own, the main session's markers and push record stay untouched.
Synthetic transcripts only. CONTEXT_HOOKS points the tests at a copy of the hooks (built and tested before install).

    .venv-cpu/bin/python -m pytest tests/test_compacted_agent.py      (or: python3 tests/test_compacted_agent.py)"""
import json, os, subprocess, sys, tempfile, time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOKS = os.environ.get("CONTEXT_HOOKS") or os.path.join(ROOT, ".claude", "hooks")
AGENT = "a0000000000000009"


def stamp(age):
    return datetime.fromtimestamp(time.time() - age, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def world(tmp, main_age=None, agent_age=None, agent_mtime_age=0):
    """A session laid out as Claude Code lays it out: main transcript and one agent's, each with a compact_boundary
    `age` seconds old (None: none)."""
    sub = os.path.join(tmp, "proj", "sess1", "subagents")
    os.makedirs(sub)
    main_t = os.path.join(tmp, "proj", "sess1.jsonl")
    for path, age in ((main_t, main_age), (os.path.join(sub, "agent-%s.jsonl" % AGENT), agent_age)):
        with open(path, "w") as f:
            f.write(json.dumps({"type": "user", "timestamp": stamp(9999), "message": {"content": "x"}}) + "\n")
            if age is not None:
                f.write(json.dumps({"type": "system", "subtype": "compact_boundary", "timestamp": stamp(age)}) + "\n")
    p = os.path.join(sub, "agent-%s.jsonl" % AGENT)
    os.utime(p, (time.time() - agent_mtime_age,) * 2)
    os.makedirs(os.path.join(tmp, ".claude"))
    return main_t


def inp(main_t, **k):
    d = {"hook_event_name": "SessionStart", "source": "compact", "session_id": "sess1", "transcript_path": main_t}
    d.update(k)
    return d


def cg():
    sys.path.insert(0, HOOKS)
    import importlib, context_guard
    return importlib.reload(context_guard)


def test_detection():
    m = cg()
    cases = [("agent just compacted, main long ago", dict(main_age=5000, agent_age=0.4), AGENT),
             ("agent just compacted, main never", dict(agent_age=0.4), AGENT),
             ("main just compacted, agent long ago", dict(main_age=0.4, agent_age=4000), None),
             ("main just compacted, agent never", dict(main_age=0.4), None),
             ("agent compacted 10 min ago, transcript stale", dict(main_age=0.4, agent_age=600, agent_mtime_age=600), None),
             ("both just now, main newer", dict(main_age=0.2, agent_age=0.6), None)]
    for name, w, want in cases:
        with tempfile.TemporaryDirectory() as tmp:
            got = m.compacted_agent(inp(world(tmp, **w)))
        assert got == want, "%s: got %r" % (name, got)
    assert m.compacted_agent({"agent_id": "x"}) == "x"
    assert m.compacted_agent({}) is None


def run(hook, tmp, d, *args):
    env = dict(os.environ, CLAUDE_PROJECT_DIR=tmp)
    r = subprocess.run([sys.executable, os.path.join(HOOKS, hook)] + list(args), input=json.dumps(d),
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_guard_agent_compaction_keeps_markers():
    with tempfile.TemporaryDirectory() as tmp:
        main_t = world(tmp, main_age=5000, agent_age=0.4)
        for m in (".compact_wanted", ".compact_checked"):
            open(os.path.join(tmp, ".claude", m), "w").close()
        out = json.loads(run("context_guard.py", tmp, inp(main_t)))["hookSpecificOutput"]["additionalContext"]
        assert "your context was just compacted" in out and "THE PLAN" not in out, out
        assert all(os.path.exists(os.path.join(tmp, ".claude", m)) for m in (".compact_wanted", ".compact_checked"))


def test_guard_main_compaction_clears_markers():
    with tempfile.TemporaryDirectory() as tmp:
        main_t = world(tmp, main_age=0.4, agent_age=5000)
        open(os.path.join(tmp, ".claude", ".compact_wanted"), "w").close()
        out = json.loads(run("context_guard.py", tmp, inp(main_t)))["hookSpecificOutput"]["additionalContext"]
        assert "this context was just compacted" in out and "THE PLAN" in out, out
        assert not os.path.exists(os.path.join(tmp, ".claude", ".compact_wanted"))


def test_push_skips_agent_compaction():
    with tempfile.TemporaryDirectory() as tmp:
        main_t = world(tmp, main_age=5000, agent_age=0.4)
        out = run("context_push.py", tmp, inp(main_t), "session")
        assert "additionalContext" not in out, out


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_"):
            f(); n += 1; print("PASS", k)
    print("%d/%d pass" % (n, n))
