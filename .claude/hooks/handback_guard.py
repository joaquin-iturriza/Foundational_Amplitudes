#!/usr/bin/env python3
"""PreToolUse(SubagentHandback), plan entry agent_redesign (the double wake): a subagent that hands its report back
with SubagentHandback wakes the main session twice (the hand-back, then the task-notification when it stops); one that
ends with its report as its final plain text wakes it once, with the report whole (agent acc18380b00739f5c, 01:05).
Prompts alone do not stop the call (a harness reminder asks for it), so it is refused here, for subagents only (a
call carrying agent_id); the main session and every other tool pass. Exit 2 with the message on stderr; exit 1
(non-blocking) on unreadable input."""
import json, sys

MSG = ("handback_guard: SubagentHandback is refused in this project (it wakes the main session twice). End your run "
       "instead with your full report as your final plain-text message, with no tool call after it: that text is "
       "delivered to your caller as your result.")


def main():
    try:
        inp = json.load(sys.stdin)
    except ValueError:
        return 1
    if not isinstance(inp, dict) or inp.get("tool_name") != "SubagentHandback" or not inp.get("agent_id"):
        return 0
    print(MSG, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
