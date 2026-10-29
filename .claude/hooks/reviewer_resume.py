#!/usr/bin/env python3
"""A blocking pillar review is re-checked by continuing the same reviewer, never by a fresh one (the user, 2026-10-29:
"Why as a line in claude md instead of making it mechanical?"; CLAUDE.md, batched pillar review).

  PostToolUse(Agent): a pillar reviewer's launch records its agent id in .claude/.review_state/<pillar>.agent.
  PreToolUse(Agent):  a launch of a pillar reviewer is refused while that pillar's lock is fresh (a review in progress,
                      or a blocking verdict whose fixes are being applied) and a recorded reviewer holds it; the
                      refusal names the agent id to continue with SendMessage. A stale lock (180 min, as
                      review_backlog.sh LOCK_TTL_MIN) lets a fresh reviewer through.
"""
import json, os, re, sys, time

ROOT = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE = os.path.join(ROOT, ".claude", ".review_state")
PILLARS = ("repo-reviewer", "notes-editor", "claudemd-keeper")
LOCK_TTL_S = 180 * 60


def main():
    try:
        ev = json.load(sys.stdin)
    except ValueError:
        return 0
    if ev.get("tool_name") != "Agent":
        return 0
    who = (ev.get("tool_input") or {}).get("subagent_type")
    if who not in PILLARS:
        return 0
    rec = os.path.join(STATE, who + ".agent")
    if ev.get("hook_event_name") == "PostToolUse":
        m = re.search(r"agentId: (\w+)", json.dumps(ev.get("tool_response")))
        if m:
            os.makedirs(STATE, exist_ok=True)
            open(rec, "w").write(m.group(1) + "\n")
        return 0
    lock = os.path.join(STATE, who + ".lock")
    if not (os.path.exists(lock) and os.path.exists(rec) and time.time() - os.path.getmtime(lock) < LOCK_TTL_S):
        return 0
    aid = open(rec).read().strip()
    sys.stderr.write(
        "reviewer_resume: the %s pillar's lock is held by the reviewer %s (a review in progress, or a blocking verdict "
        "whose fixes are being applied). Continue it instead of launching a new one: SendMessage to %s, naming the fix "
        "commits. A fresh reviewer goes through once the lock is released (advance) or stale (180 min).\n"
        % (who, aid, aid))
    return 2


if __name__ == "__main__":
    sys.exit(main())
