#!/usr/bin/env python3
"""PostToolUse(Agent|SendMessage), main session only (agent redesign D5). After the main session launches or resumes
an agent (each wake follows a hand-back), every running entry's owner has its context read from its transcript
(tools/plan.py agent_context): recorded on the entry (plan owner --context) when it moved by RECORD_STEP tokens or
crossed the threshold, so `plan next` says close <agent> at 300k or more; a compaction boundary inside an owner's
transcript is recorded once as a finding on its entry (its brief or scope was too large). Notes only, never a refusal;
Python, so a crash exits 1 and blocks nothing. Env for tests: the PLAN_* variables of tools/plan.py."""
import json, os, subprocess, sys

ROOT = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
RECORD_STEP = 10000


def plan_cli(*args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "tools", "plan.py")] + list(args),
                          capture_output=True, text=True, timeout=120)


def main():
    try:
        inp = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return 0
    if not isinstance(inp, dict) or inp.get("agent_id"):          # a subagent's own call: not the main session's
        return 0
    import plan
    d, st = plan.load()
    notes = []
    for e in d["entries"]:
        o = plan.owner_of(e)
        if e.get("status") != "running" or not o or o.get("state") == "closed":
            continue
        c = plan.agent_context(o["agent"])
        if not c or c["tokens"] is None:
            continue
        was = o.get("context") or 0
        if abs(c["tokens"] - was) >= RECORD_STEP or (c["tokens"] >= plan.CONTEXT_CLOSE) != (was >= plan.CONTEXT_CLOSE):
            plan_cli("owner", e["id"], "--context", str(c["tokens"]))
        if c["tokens"] >= plan.CONTEXT_CLOSE:
            notes.append("%s's owner %s is at %d tokens of context: ask it to write its hand-off (plan handoff %s "
                         "...) and end; a fresh owner starts from it" % (e["id"], o["agent"], c["tokens"], e["id"]))
        seen = {f.get("text") for f in e.get("findings") or []}
        for ts in c["compactions"]:
            text = "a compaction inside the turn of owner %s at %s: its brief or scope was too large; narrow the next " \
                   "brief" % (o["agent"], ts)
            if text not in seen:
                plan_cli("finding", e["id"], "--text", text)
                notes.append("%s: %s" % (e["id"], text))
    if notes:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse",
                                                 "additionalContext": "owner_context: " + "; ".join(notes)}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
