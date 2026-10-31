#!/usr/bin/env python3
"""PreToolUse (Monitor, Bash): nobody waits on jobs. While its jobs run an owner records them, writes where the
work stands and ends its turn; the supervisor's event resumes it (SendMessage). 2026-10-30: the generation owner held
a background `site poll` loop with a Monitor on it for about an hour, waking every few minutes, and kept it after it
had closed. The watch is the supervisor and one event waiter (~/work/CLAUDE.md), run by the main session; anything
else is refused for everyone (Joaquin, 2026-10-30: "Why would we allow the main session to do it but not the
agents?"): a Monitor, and a background command that polls job state or sleeps in a loop. An agent's other background
command goes through tools/detach.py (2026-10-31: the detacher exists so that an agent's background work ending after
the agent stopped wakes its owner through a site event, never the main session through a task notice); the main
session's own background work and its watch are unchanged."""
import json
import os
import re
import shlex
import subprocess
import sys

POLL = re.compile(r"\bsite\s+(?:--\S+\s+)*(?:poll|runs|logs|status|events)\b|\bsqueue\b|\bcondor_q\b")
WATCH = re.compile(r"^\s*(?:cd\s+\S+\s*(?:&&|;)\s*)?site\s+(?:--\S+\s+)*(?:events\s+--wait|supervise)\b[^;&|]*$")
DETACH = re.compile(r"^\s*(?:cd\s+\S+\s*(?:&&|;)\s*)?(?:python3?\s+)?(?:\S*/)?tools/detach\.py\b")
LOOP = re.compile(r"\b(?:for|while|until)\b[^;]*;\s*do\b|\bwatch\s|\bsleep\s+\d")


def entry_of(agent):
    """The plan entry this agent owns (plan owners), or <entry>."""
    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        out = subprocess.run([sys.executable, os.path.join(root, "tools", "plan.py"), "owners"], capture_output=True,
                             text=True, timeout=20, cwd=root).stdout
        return next((x["entry"] for x in json.loads(out.strip().splitlines()[-1]) if x.get("owner") == agent), "<entry>")
    except Exception:
        return "<entry>"


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
    elif tool == "Bash" and ti.get("run_in_background") and not WATCH.match(cmd) \
            and (POLL.search(cmd) or LOOP.search(cmd)):
        what = "a background command that polls or loops"
    elif tool == "Bash" and ti.get("run_in_background") and agent and not DETACH.match(cmd):
        print("wait_guard: an agent's background command goes through the detacher, so that its end resumes you "
              "through a site event and never wakes the main session. Run instead (not in the background):\n"
              "  python3 tools/detach.py --entry %s -- bash -c %s" % (entry_of(inp.get("agent_id")), shlex.quote(cmd)),
              file=sys.stderr)
        return 2
    else:
        return 0
    print("wait_guard: %s is refused: nobody waits on jobs. %s" % (what,
          "Jobs: record them (plan run), write where the work stands and end your turn; the "
          "supervisor's event resumes you with SendMessage." if agent else "The watch is `site supervise` and one "
          "`site events --wait` in the background; an event wakes you, nothing else waits."), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
