#!/usr/bin/env python3
"""PreToolUse (Monitor, Bash): an agent never waits on jobs. While its jobs run an owner records them, writes where the
work stands and ends its turn; the supervisor's event resumes it (SendMessage). 2026-10-30: the generation owner held
a background `site poll` loop with a Monitor on it for about an hour, waking every few minutes, and kept it after it
had closed. Refused for an agent (the main session's own waiter, `site events --wait`, is not an agent's call):
a Monitor, and a background command that polls job state or sleeps in a loop."""
import json
import re
import sys

POLL = re.compile(r"\bsite\s+(?:--\S+\s+)*(?:poll|runs|logs|status|events)\b|\bsqueue\b|\bcondor_q\b")
LOOP = re.compile(r"\b(?:for|while|until)\b[^;]*;\s*do\b|\bwatch\s|\bsleep\s+\d")


def main():
    try:
        inp = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return 0
    if not isinstance(inp, dict) or not inp.get("agent_id"):
        return 0
    tool = inp.get("tool_name") or ""
    ti = inp.get("tool_input") or {}
    cmd = ti.get("command") or ""
    if tool == "Monitor":
        what = "a Monitor"
    elif tool == "Bash" and ti.get("run_in_background") and (POLL.search(cmd) or LOOP.search(cmd)):
        what = "a background command that polls or loops"
    else:
        return 0
    print("owner_wait_guard: %s is refused for an agent. Do not wait on jobs: record them (plan run), write where the "
          "work stands (plan evidence / set-next) and end your turn; the supervisor's event resumes you with "
          "SendMessage. A long CPU analysis goes to a site job, not a background loop here." % what, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
