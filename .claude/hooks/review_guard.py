#!/usr/bin/env python3
"""One review guard (agent redesign, step B3): lock_guard.py and reviewer_resume.py in one script, so one reviewer
works one backlog.

  PreToolUse(Bash)    a command that runs `review_backlog.sh begin <name>` or `advance <name>` is refused (exit 2)
                      unless the tool call comes from the subagent <name> itself (the hook input's agent_type; the
                      main session's carries none). The raw text is read minus what is proven data (cmdparse.py): a
                      commit message, a grep pattern, a heredoc body written by cat is not a call. Every such call is
                      logged with its caller to .claude/.review_state/lock_calls.log. The user's override is a `!`
                      command, which runs no hook (the user, 2026-10-23: "you are not allowed to manually release it").
  PostToolUse(Agent)  a pillar reviewer's launch records its agent id in .claude/.review_state/<pillar>.agent.
  PreToolUse(Agent)   a launch of a pillar reviewer is refused while that pillar's lock is fresh (a review in progress,
                      or a blocking verdict whose fixes are being applied) and a recorded reviewer holds it; the
                      refusal names the agent to continue with SendMessage (the user, 2026-10-29: "Why as a line in
                      claude md instead of making it mechanical?"). A stale lock (180 min, review_backlog.sh
                      LOCK_TTL_MIN) lets a fresh reviewer through.
Python on purpose: a crash exits 1, which does not block."""
import datetime, json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmdparse  # noqa: E402

ROOT = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE = os.path.join(ROOT, ".claude", ".review_state")
LOG = os.path.join(STATE, "lock_calls.log")
PILLARS = ("repo-reviewer", "notes-editor", "claudemd-keeper")
LOCK_TTL_S = 180 * 60
PAT = re.compile(r"review_backlog\.sh\s+(begin|advance)\b(?:\s+([A-Za-z_-]+))?")
DATA_PROGS = {"grep", "egrep", "fgrep", "rg", "git"}     # their arguments are patterns and messages
DATA_FEEDERS = {"cat", "tee", "git"}                     # a heredoc body fed to them is text written or committed
# an interpreter's program (a heredoc body fed to python, python -c '...') runs a shell command only through one of
# these; without one, a string naming the script in it is data (a verdict text written to a JSON file)
RUNNER = re.compile(r"os\.system|subprocess|popen|\bexec[lv]p?e?\(|\bspawn|check_(?:call|output)|\bsystem\s*\(|`")


def lock_calls(cmd):
    """The (verb, reviewer) of every review_backlog.sh begin|advance in the command, read on the raw text (a python -c
    os.system, a script echoed then run, bash -lc, "$(...)" all count) minus the contexts that are data: the arguments
    of grep / rg / git, a heredoc body fed to cat, tee or git, and an interpreter's program (heredoc or -c) that calls
    no process runner (RUNNER). A command substitution in those still runs, so it is read too."""
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


def bash(inp):
    cmd = (inp.get("tool_input") or {}).get("command", "") or ""
    calls = lock_calls(cmd)
    if not calls:
        return 0
    caller = inp.get("agent_type") or ""
    os.makedirs(STATE, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} caller={caller or 'main'} "
                f"calls={calls} keys={sorted(inp)}\n")
    for verb, target in calls:
        if not target or caller != target:
            print(f"BLOCKED by review_guard: '{verb} {target or '?'}' may only be run by the {target or 'named'} "
                  f"reviewer subagent itself (this call comes from {caller or 'the main session'}). Spawn that reviewer "
                  f"on its whole backlog; it takes and releases its own lock. The user's override is a '!' command.",
                  file=sys.stderr)
            return 2
    return 0


def agent(inp):
    who = (inp.get("tool_input") or {}).get("subagent_type")
    if who not in PILLARS:
        return 0
    rec = os.path.join(STATE, who + ".agent")
    if inp.get("hook_event_name") == "PostToolUse":
        m = re.search(r"agentId: (\w+)", json.dumps(inp.get("tool_response")))
        if m:
            os.makedirs(STATE, exist_ok=True)
            open(rec, "w").write(m.group(1) + "\n")
        return 0
    lock = os.path.join(STATE, who + ".lock")
    if not (os.path.exists(lock) and os.path.exists(rec) and time.time() - os.path.getmtime(lock) < LOCK_TTL_S):
        return 0
    aid = open(rec).read().strip()
    sys.stderr.write(
        "review_guard: the %s pillar's lock is held by the reviewer %s (a review in progress, or a blocking verdict "
        "whose fixes are being applied). Continue it instead of launching a new one: SendMessage to %s, naming the fix "
        "commits. A fresh reviewer goes through once the lock is released (advance) or stale (180 min).\n"
        % (who, aid, aid))
    return 2


def main():
    try:
        inp = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return 0
    tool = inp.get("tool_name")
    if tool == "Agent":
        return agent(inp)
    if tool == "Bash" or (inp.get("tool_input") or {}).get("command") is not None:
        return bash(inp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
