#!/usr/bin/env python3
"""PreToolUse(Bash): refuse a fixed wait. A `sleep N` (N >= 3 s) anywhere in a command, not inside an until/while loop
that waits on a condition, burns the user's time whatever happens. Claude Code's own block catches only a command that
starts with a sleep; 2026-10-25 a `...; wait; sleep 40; tail log` slipped through it. Wait on the condition instead
(`until grep -q ... log; do sleep 5; done`) or run in the background."""
import json, re, sys

try:
    cmd = json.load(sys.stdin).get("tool_input", {}).get("command", "")
except ValueError:
    sys.exit(0)
# drop the bodies of until/while loops: a short sleep there is a poll on a condition, which is fine
body = re.sub(r"\b(until|while)\b.*?\bdone\b", " ", cmd, flags=re.S)
for m in re.finditer(r"\bsleep\s+(\d+(?:\.\d+)?)", body):
    if float(m.group(1)) >= 3:
        print(f"sleep_guard: a fixed `sleep {m.group(1)}` outside a condition loop. Wait on the condition "
              "(until <check>; do sleep 5; done) or run the command in the background.", file=sys.stderr)
        sys.exit(2)
sys.exit(0)
