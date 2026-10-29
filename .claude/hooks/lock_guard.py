#!/usr/bin/env python3
"""Only a pillar's own reviewer may take or release its review lock (the user, 2026-10-23: "you are not allowed to
manually release it ... you just kept sidestepping the review by manually releasing it without running the reviewer").

PreToolUse(Bash): a command that runs `review_backlog.sh begin <name>` or `review_backlog.sh advance <name>` is refused
(exit 2) unless the tool call comes from the subagent <name> itself. The raw text is read, minus what is proven data
(cmdparse.py): a commit message, a grep pattern or a heredoc body written by cat is not a call. The hook
input of a tool call made inside a subagent carries agent_type (the subagent's name); the main session's carries none. The user's own override is a `!`
command, which runs no hook. Every such call is logged with its caller to .claude/.review_state/lock_calls.log.
Python on purpose: a crash here exits 1, which does not block (hook_syntax_guard.py).
"""
import datetime, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmdparse  # noqa: E402

ROOT = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG = os.path.join(ROOT, ".claude", ".review_state", "lock_calls.log")
PAT = re.compile(r"review_backlog\.sh\s+(begin|advance)\b(?:\s+([A-Za-z_-]+))?")


DATA_PROGS = {"grep", "egrep", "fgrep", "rg", "git"}     # their arguments are patterns and messages
DATA_FEEDERS = {"cat", "tee", "git"}                     # a heredoc body fed to them is text written or committed
# an interpreter's program (a heredoc body fed to python, python -c '...') runs a shell command only through one of
# these; without one, a string naming the script in it is data (a verdict text written to a JSON file)
RUNNER = re.compile(r"os\.system|subprocess|popen|\bexec[lv]p?e?\(|\bspawn|check_(?:call|output)|\bsystem\s*\(|`")


def lock_calls(cmd):
    """The (verb, reviewer) of every review_backlog.sh begin|advance in the command, read on the raw text (a python -c
    os.system, a script echoed then run, bash -lc, "$(...)" all count) minus the contexts that are data: the arguments
    of grep / rg / git (a commit message, a pattern), a heredoc body fed to cat, tee or git, and an interpreter's
    program (heredoc or -c) that calls no process runner (RUNNER). A command substitution in those still runs, so it
    is read too."""
    top, bodies = cmdparse.heredocs(cmd)
    text = []
    for line, b in bodies:
        f = cmdparse.feeder(line)
        if cmdparse.INTERP.match(f or ""):
            if RUNNER.search(b):
                text.append(b)
        elif f not in DATA_FEEDERS:
            text.append(b)
        elif not cmdparse.HEREDOC.search(line).group(2):           # <<EOF unquoted: its $(...) run
            text += cmdparse.substitutions(b, in_heredoc=True)
    segs = cmdparse.segments(top)
    if segs is None:
        text.append(top)
    else:
        for seg in segs:
            p, joined = cmdparse.prog(cmdparse.words(seg)), " ".join(seg)
            if p in DATA_PROGS or (cmdparse.INTERP.match(p) and not RUNNER.search(joined)):
                continue
            text.append(joined)
        text += cmdparse.substitutions(top)
    return PAT.findall("\n".join(text))


def main():
    try:
        inp = json.loads(sys.stdin.read() or "{}")
    except Exception:
        return 0
    cmd = (inp.get("tool_input") or {}).get("command", "") or ""
    calls = lock_calls(cmd)
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
