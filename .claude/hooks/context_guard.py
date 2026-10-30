#!/usr/bin/env python3
"""Context guard (the user's rule, 2026-10-05): when the context fills, the notes are written and then the session
compacts, with nothing in between, so a long autonomous run does not drift.

  PreCompact    (trigger "auto") the notes not flushed since the last compaction: refuse it (exit 2) and record
                that a compaction is wanted (WANTED). Flushed, or a manual /compact: let it through.
  Stop          a compaction is wanted and the notes are not flushed: block, and say what to do.
  PostToolUse   the same condition: the instruction as additional context, so it is seen mid-turn too.
  SessionStart  every source: print the plan (tools/plan.py show --compact); after a compaction also clear both
                markers and say that the summary is not the plan.

When auto-compaction fires is Claude Code's call (env CLAUDE_CODE_AUTO_COMPACT_WINDOW in settings.json); a hook
cannot start one, only hold it. Held, it is retried on the next auto-compaction check, which then finds the flush.
"""
import json, os, sys

REPO = os.environ.get("CLAUDE_PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FLUSHED = os.path.join(REPO, ".claude", ".notes_flushed")
WANTED = os.path.join(REPO, ".claude", ".compact_wanted")

INSTR = ("CONTEXT GUARD: Claude Code wants to compact the context and is held until the notes are written. Do this now, "
         "before anything else: update docs/results.tex with everything needed to continue (the state of every running "
         "batch: sites, counts, waiters; decisions taken since the last compaction; results not yet written up; the "
         "plan's next steps in the transfer study's hand-off, the 'Plan to finish' item); commit; then run "
         "`touch .claude/.notes_flushed` and end the turn. Compaction then goes through.")
AFTER = ("CONTEXT GUARD: this context was just compacted. The summary above is a lossy record of the CONVERSATION (what "
         "was in flight), not of the plan: work only from the plan printed below, not from the summary's "
         "pending list. CLAUDE.md's ground rules hold (0: start every message with 'Joaquin'; 0b: never narrow the scope). "
         "Re-check the running batches with `site` before acting on any number in the summary.")


def plan_text():
    """The plan itself (tools/plan.py show --compact): the only source of truth, printed so no summary stands
    between the fresh context and it (setup redesign, D121)."""
    import subprocess
    try:
        r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "plan.py"), "show", "--compact"],
                           capture_output=True, text=True, timeout=60)
        return "THE PLAN (plan/plan.yaml; launch a step agent for each entry `plan next` lists, now):\n" + r.stdout + r.stderr
    except Exception as e:                      # noqa: BLE001
        return "THE PLAN could not be printed (%r): run python3 tools/plan.py show now." % e


def notes_at():
    """Context size at which the notes are asked for: 50k below the auto-compaction window (Joaquin, 2026-10-30:
    "Reduce the threshold of compaction to 400k, and the notes updating a bit lower")."""
    try:
        return int(os.environ.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") or 400000) - 50000
    except ValueError:
        return 350000


def context_tokens(path):
    """The session's current context: the last assistant call's input, cache read and cache write, from the
    transcript's tail. Unreadable: 0."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - (1 << 20)))
            lines = f.read().split(b"\n")
    except (OSError, TypeError):
        return 0
    for line in reversed(lines):
        if b'"usage"' not in line:
            continue
        try:
            u = json.loads(line)["message"]["usage"]
        except (ValueError, KeyError, TypeError):
            continue
        return (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                + u.get("cache_creation_input_tokens", 0))
    return 0


def _clear(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def main():
    try:
        inp = json.load(sys.stdin)
    except ValueError:
        return 0
    ev = inp.get("hook_event_name", "")
    if ev == "PreCompact":
        if inp.get("trigger") != "auto" or os.path.exists(FLUSHED):
            return 0
        open(WANTED, "w").close()
        sys.stderr.write(INSTR)
        return 2
    if ev == "SessionStart":
        ctx = plan_text()                     # every source (startup, resume, clear, compact) starts from the plan
        if inp.get("source") == "compact":
            _clear(FLUSHED)
            _clear(WANTED)
            ctx = AFTER + "\n\n" + ctx
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ctx}}))
        return 0
    if not os.path.exists(WANTED) and not os.path.exists(FLUSHED) and not inp.get("agent_id") \
            and context_tokens(inp.get("transcript_path")) >= notes_at():
        open(WANTED, "w").close()          # the notes are asked for before compaction, not only when it fires
    if not os.path.exists(WANTED) or os.path.exists(FLUSHED):
        return 0
    if ev == "Stop":
        if inp.get("stop_hook_active"):
            return 0                     # already blocked on this stop: do not loop
        print(json.dumps({"decision": "block", "reason": INSTR}))
    elif ev == "PostToolUse":
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": INSTR}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
