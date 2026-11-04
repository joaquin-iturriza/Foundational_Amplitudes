#!/usr/bin/env python3
"""UserPromptSubmit: absorb the second wake of an agent end (agent_redesign next 15, D279).

Each background subagent's end gives the main session two signals: its SubagentHandback message (the report) and the
harness's <task-notification> "Agent ... finished", whose <result> says the report "was delivered to you as a message
from <id> (its SubagentHandback call)". Mid-turn the notification is folded into the running turn at no cost; when it
arrives after the turn ended it starts a turn of its own, and in 4 of 32 such cases since 2026-10-31 the model was
called on it at the full context (main_turns.py agent_done).

This hook blocks that prompt, so no model call is made on it, only when all hold:
  - the prompt is a <task-notification> whose task id is an agent id (a + 16 hex) and whose status is completed;
  - its result says the report was delivered through SubagentHandback (nothing in it is new);
  - the prompt is that one notification and nothing else (no second notification, no reminder after it);
  - the transcript holds a hand-back message from that same agent (the report did reach this session).
Anything else (a notification with the report inline, an agent waiting on its own work, a failed or killed agent,
a background command, typed text) passes untouched. A crash exits 1 (non-blocking: the prompt goes through).
    python3 notification_absorb.py            (hook input on stdin)
    python3 notification_absorb.py --check <transcript.jsonl> <prompt file>    the verdict, for tests and replays
"""
import json, re, sys

AGENT = re.compile(r"<task-id>(a[0-9a-f]{16})</task-id>")
DELIVERED = re.compile(r"report was delivered to you as a message from \"?(a[0-9a-f]{16})\"? \(its SubagentHandback call\)")


def handed_back(transcript, agent):
    """True when the transcript holds a hand-back message from `agent` (a queued message or a turn-starting one)."""
    tag = 'agent-message from=\\"%s\\"' % agent      # as written inside a JSON string
    tag_plain = 'agent-message from="%s"' % agent
    try:
        with open(transcript, errors="replace") as f:
            for line in f:
                if (tag in line or tag_plain in line) and "Subagent hand-back" in line:
                    return True
    except OSError:
        return False
    return False


def verdict(prompt, transcript):
    """(block, why) for one prompt."""
    s = prompt or ""
    i = s.find("<task-notification>")
    if i < 0 or i > 1500:          # the harness's notification preamble may come first, nothing else
        return False, "not a task-notification"
    s = s[i:]
    end = s.find("</task-notification>")
    if end < 0 or s.count("<task-notification>") != 1 or s[end + len("</task-notification>"):].strip():
        return False, "more than one notification, or text after it"      # anything else in it may be new
    s = s[:end]
    m = AGENT.search(s[:300])
    if not m:
        return False, "not an agent's notification"
    agent = m.group(1)
    if "<status>completed</status>" not in s:
        return False, "status is not completed"
    d = DELIVERED.search(s)
    if not d or d.group(1) != agent:
        return False, "the result is not a SubagentHandback delivery"
    if not transcript or not handed_back(transcript, agent):
        return False, "no hand-back from %s in the transcript" % agent
    return True, agent


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--check":
        with open(sys.argv[3]) as f:
            print(json.dumps(dict(zip(("block", "why"), verdict(f.read(), sys.argv[2])))))
        return 0
    d = json.load(sys.stdin)
    if d.get("hook_event_name", "UserPromptSubmit") != "UserPromptSubmit":
        return 0
    block, why = verdict(str(d.get("prompt") or ""), d.get("transcript_path"))
    if block:
        print(json.dumps({"decision": "block",
                          "reason": "agent %s finished; its report was already delivered by its hand-back" % why}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:      # never block on a crash
        print("notification_absorb.py: %s" % e, file=sys.stderr)
        sys.exit(1)
