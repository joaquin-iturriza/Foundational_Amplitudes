#!/usr/bin/env python3
"""Context guard (the user's rule, 2026-10-05): before the context window fills, everything the work needs goes into the
notes, and then the session compacts, so a long autonomous run does not drift.

  Stop          past FLUSH_AT of the window and the notes not flushed in this context: block, and say what to do.
  PostToolUse   the same condition: a reminder as additional context, so it is seen mid-turn too.
  SessionStart  (source "compact"): point the fresh context at the notes' plan and standing rules.

Compaction itself is Claude Code's auto-compact, set to COMPACT_AT in settings.json
(env CLAUDE_CODE_AUTOCOMPACT_PCT_OVERRIDE): a hook cannot start one. Context use is the last main-chain assistant
message's input + cache-read + cache-creation tokens. A flush is recorded by `touch .claude/.notes_flushed`; the
marker is cleared once use falls below RESET_AT (after a compaction), so the next fill needs a new flush.
"""
import json, os, sys

WINDOW = int(os.environ.get("FA_CONTEXT_WINDOW", "1000000"))
FLUSH_AT, RESET_AT, COMPACT_AT = 0.50, 0.30, 0.60
REPO = os.environ.get("CLAUDE_PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MARK = os.path.join(REPO, ".claude", ".notes_flushed")

INSTR = (f"CONTEXT GUARD: the context window is at {{pct:.0f}}% (flush at {FLUSH_AT:.0%}, auto-compaction at "
         f"{COMPACT_AT:.0%}). Before anything else: update docs/results.tex with everything needed to continue: the state "
         "of every running batch (sites, counts, waiters), decisions taken since the last flush, results not yet written "
         "up, and the plan's next steps in the transfer study's hand-off (the 'Plan to finish' item); commit; then run "
         "`touch .claude/.notes_flushed`. Then carry on with the plan; auto-compaction follows by itself.")
AFTER = ("CONTEXT GUARD: this context was just compacted. Before continuing, read in docs/results.tex the transfer "
         "study's hand-off (sec:ladder-open), in particular the 'Plan to finish' item, and CLAUDE.md's ground rules "
         "(0: start every message with 'Joaquin'; 0b: never narrow the scope). Re-check the running batches with `site` "
         "before acting on any number in the summary.")


def usage(path):
    last = None
    try:
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("type") == "assistant" and not d.get("isSidechain"):
                    u = (d.get("message") or {}).get("usage")
                    if u:
                        last = u
    except OSError:
        return None
    if not last:
        return None
    return sum(int(last.get(k) or 0) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))


def main():
    try:
        inp = json.load(sys.stdin)
    except ValueError:
        return 0
    ev = inp.get("hook_event_name", "")
    if ev == "SessionStart":
        if inp.get("source") == "compact":
            if os.path.exists(MARK):
                os.remove(MARK)
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": AFTER}}))
        return 0
    n = usage(inp.get("transcript_path", ""))
    if n is None:
        return 0
    frac = n / WINDOW
    if frac < RESET_AT and os.path.exists(MARK):
        os.remove(MARK)
    if frac < FLUSH_AT or os.path.exists(MARK):
        return 0
    msg = INSTR.format(pct=100 * frac)
    if ev == "Stop":
        if inp.get("stop_hook_active"):
            return 0                     # already blocked on this stop: do not loop
        print(json.dumps({"decision": "block", "reason": msg}))
    elif ev == "PostToolUse":
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": msg}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
