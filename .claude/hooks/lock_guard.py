#!/usr/bin/env python3
"""Only a pillar's own reviewer may take or release its review lock (the user, 2026-10-23: "you are not allowed to
manually release it ... you just kept sidestepping the review by manually releasing it without running the reviewer").

PreToolUse(Bash): a command that runs `review_backlog.sh begin <name>` or `review_backlog.sh advance <name>` is refused
(exit 2) unless the tool call comes from the subagent <name> itself. The hook input of a tool call made inside a
subagent carries agent_type (the subagent's name); the main session's carries none. The user's own override is a `!`
command, which runs no hook. Every such call is logged with its caller to .claude/.review_state/lock_calls.log.
Python on purpose: a crash here exits 1, which does not block (hook_syntax_guard.py).
"""
import datetime, json, os, re, sys

ROOT = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG = os.path.join(ROOT, ".claude", ".review_state", "lock_calls.log")
PAT = re.compile(r"review_backlog\.sh\s+(begin|advance)\b(?:\s+([A-Za-z_-]+))?")


def main():
    try:
        inp = json.loads(sys.stdin.read() or "{}")
    except Exception:
        return 0
    cmd = (inp.get("tool_input") or {}).get("command", "") or ""
    calls = PAT.findall(cmd)
    if not calls:
        return 0
    caller = inp.get("agent_type") or ""
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} caller={caller or 'main'} "
                f"calls={calls} keys={sorted(inp)}\n")
    for verb, target in calls:
        if not target or caller != target:
            print(f"BLOCKED by lock_guard: '{verb} {target or '?'}' may only be run by the {target or 'named'} reviewer "
                  f"subagent itself (this call comes from {caller or 'the main session'}). Spawn that reviewer on its "
                  f"whole backlog; it takes and releases its own lock. The user's override is a '!' command.",
                  file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
