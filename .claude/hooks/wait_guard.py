#!/usr/bin/env python3
"""PreToolUse (Monitor, Bash): nobody waits on jobs. While its jobs run an owner records them, writes where the
work stands and ends its turn; the supervisor's event resumes it (SendMessage). 2026-10-30: the generation owner held
a background `site poll` loop with a Monitor on it for about an hour, waking every few minutes, and kept it after it
had closed. The watch is the supervisor and one event waiter (~/work/CLAUDE.md), run by the main session; anything
else is refused for everyone (Joaquin, 2026-10-30: "Why would we allow the main session to do it but not the
agents?"): a Monitor, and a background command that polls job state or sleeps in a loop."""
import json
import re
import sys

POLL = re.compile(r"\bsite\s+(?:--\S+\s+)*(?:poll|runs|logs|status|events)\b|\bsqueue\b|\bcondor_q\b")
WATCH = re.compile(r"^\s*site\s+(?:--\S+\s+)*(?:events\s+--wait|supervise)\b[^;&|]*$")
LOOP = re.compile(r"\b(?:for|while|until)\b[^;]*;\s*do\b|\bwatch\s|\bsleep\s+\d")


def main():
    try:
        inp = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return 0
    if not isinstance(inp, dict):
        return 0
    agent = bool(inp.get("agent_id"))
    tool = inp.get("tool_name") or ""
    ti = inp.get("tool_input") or {}
    cmd = ti.get("command") or ""
    if tool == "Monitor":
        what = "a Monitor"
    elif tool == "Bash" and ti.get("run_in_background") and not (WATCH.match(cmd) and not agent) \
            and (POLL.search(cmd) or LOOP.search(cmd)):
        what = "a background command that polls or loops"
    else:
        return 0
    print("wait_guard: %s is refused: nobody waits on jobs. %s" % (what,
          "Record them (plan run), write where the work stands (plan evidence / set-next) and end your turn; the "
          "supervisor's event resumes you with SendMessage." if agent else "The watch is `site supervise` and one "
          "`site events --wait` in the background; an event wakes you, nothing else waits."), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
