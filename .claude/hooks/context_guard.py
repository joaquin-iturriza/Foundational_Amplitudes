#!/usr/bin/env python3
"""Context guard (the user's rule, 2026-10-05): when the context fills, the notes are written and then the session
compacts, with nothing in between, so a long autonomous run does not drift.

  PreCompact    (trigger "auto") the notes not flushed since the last compaction: refuse it (exit 2) and record
                that a compaction is wanted (WANTED). Flushed, or a manual /compact: let it through.
  Stop          a compaction is wanted and the notes are not flushed: block, and say what to do.
  PostToolUse   the same condition: the instruction as additional context, so it is seen mid-turn too.
  SessionStart  (source "compact"): clear both markers and point the fresh context at the notes' plan and rules.

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
AFTER = ("CONTEXT GUARD: this context was just compacted. Before continuing, read in docs/results.tex the transfer "
         "study's hand-off (sec:ladder-open), in particular the 'Plan to finish' item, and CLAUDE.md's ground rules "
         "(0: start every message with 'Joaquin'; 0b: never narrow the scope). Re-check the running batches with `site` "
         "before acting on any number in the summary.")


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
        if inp.get("source") == "compact":
            _clear(FLUSHED)
            _clear(WANTED)
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": AFTER}}))
        return 0
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
