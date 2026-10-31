#!/usr/bin/env python3
"""PostToolUse (every tool), owner agents only (agent_redesign next 10, D224). An owner can pass CONTEXT_CLOSE (170k,
tools/plan.py) and the 200k compaction window inside one turn, and `plan next` reads an owner's context only between
turns (owner_context.py). So after each tool call of a subagent, its own context is read from the tail of its own
transcript (the last assistant call's input + cache read + cache write); at CONTEXT_CLOSE or more, and when that agent
is the recorded, not closed owner of a plan entry, the agent is told to write its hand-off and end.

Who is who: a subagent's call carries `agent_id` in the hook input; the main session's never does, so the main session
is never told anything. A subagent that owns no entry (reviewers, mechanical workers, lookups) is silent too. Once the
owner writes `plan handoff <entry>`, its owner record is closed and the note stops, so it can end cleanly.

Transcript: `agent_transcript_path` when the input carries it, else <dir of transcript_path>/<session_id>/subagents/
agent-<agent_id>.jsonl (as dispatch.py finds it), else tools/plan.py agent_file().
A live-test override: a file `.owner_handoff_test` beside this script with lines "<agent id> <tokens>" lowers the
threshold for that agent only. Notes only, never a refusal; any error exits 0 silently."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(HERE))
OVERRIDE = os.path.join(HERE, ".owner_handoff_test")


def context_of(lines):
    """Context at the last model call in these transcript lines (bytes, newest last): 0 when none."""
    for line in reversed(lines):
        if b'"usage"' not in line or b'"assistant"' not in line:
            continue
        try:
            m = json.loads(line)
            if m.get("type") != "assistant":
                continue
            u = m["message"]["usage"]
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        return sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_read_input_tokens",
                                                 "cache_creation_input_tokens"))
    return 0


def tail_context(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - (1 << 20)))
            return context_of(f.read().split(b"\n"))
    except (OSError, TypeError):
        return 0


def transcript(inp, plan_mod=None):
    p = inp.get("agent_transcript_path")
    if p and os.path.exists(p):
        return p
    t, s, a = inp.get("transcript_path"), inp.get("session_id"), inp.get("agent_id")
    if t and s:
        p = os.path.join(os.path.dirname(t), s, "subagents", "agent-%s.jsonl" % a)
        if os.path.exists(p):
            return p
    return plan_mod.agent_file(a) if plan_mod else None


def threshold(agent, default):
    try:
        for line in open(OVERRIDE):
            w = line.split()
            if len(w) == 2 and w[0] == agent:
                return int(w[1])
    except (OSError, ValueError):
        pass
    return default


def main():
    try:
        inp = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return 0
    if not isinstance(inp, dict):
        return 0
    agent = inp.get("agent_id")
    if not agent:                                   # the main session: never
        return 0
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import plan
    path = transcript(inp, plan)
    tokens = tail_context(path) if path else 0
    at = threshold(agent, plan.CONTEXT_CLOSE)
    if tokens < at:
        return 0
    d, _ = plan.load()
    owned = [e["id"] for e in d.get("entries") or []
             if (plan.owner_of(e) or {}).get("agent") == agent and e["owner"].get("state") != "closed"]
    if not owned:                                   # not an owner, or its hand-off is written
        return 0
    eid = owned[0]
    msg = ("owner_handoff: your context is %d tokens, at or past the owner close line %d (CONTEXT_CLOSE, tools/plan.py), "
           "and the 200k window compacts mid-turn. Start nothing new: commit what is finished (named paths), record "
           "what you have (plan evidence / finding), then write `python3 tools/plan.py handoff %s --state ... "
           "--decisions ... --ruled-out ... --running ... --files ...` and end with your final message now; a fresh "
           "owner continues from the hand-off." % (tokens, at, eid))
    if len(owned) > 1:
        msg += " (you also own: %s; hand off each)" % ", ".join(owned[1:])
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": msg}}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                               # a note hook never blocks or crashes a turn
        sys.exit(0)
