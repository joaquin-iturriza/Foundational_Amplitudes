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
AFTER = ("CONTEXT GUARD: this context was just compacted. The summary above is a lossy record of the CONVERSATION (what "
         "was in flight), not of the plan: work only from the plan and the items printed below, not from the summary's "
         "pending list. CLAUDE.md's ground rules hold (0: start every message with 'Joaquin'; 0b: never narrow the scope). "
         "Re-check the running batches with `site` before acting on any number in the summary.")


def plan_text():
    """The open items of the hand-off (sec:ladder-open), each top-level item's first 600 characters, and the active and
    held work items: printed into the fresh context, since a pointer to them was skipped (2026-10-25: the summary listed
    only the pp generation and the plan's other items sat untouched for hours)."""
    import re, subprocess
    out = []
    try:
        s = open(os.path.join(REPO, "docs", "results.tex")).read()
        i = s.index("\\label{sec:ladder-open}")
        body, depth, items, cur = s[i:], 0, [], None
        for line in body.splitlines()[1:]:
            st = line.strip()
            if st.startswith("\\begin{itemize}"):
                depth += 1
                if depth == 1:
                    continue
            if st.startswith("\\end{itemize}"):
                depth -= 1
                if depth == 0:
                    break
            if depth == 1 and st.startswith("\\item"):
                cur = [st]; items.append(cur)
            elif cur is not None:
                cur.append(st)
        out.append("THE PLAN (docs/results.tex, sec:ladder-open), every open item, first lines:")
        for it in items:
            t = re.sub(r"\s+", " ", " ".join(it))
            out.append("  - " + (t[:600] + " ..." if len(t) > 600 else t))
    except (OSError, ValueError) as e:
        out.append("THE PLAN could not be read (%r): read docs/results.tex sec:ladder-open now." % e)
    try:
        r = subprocess.run(["site", "items", "--project", "FA"], capture_output=True, text=True, timeout=60)
        out.append("WORK ITEMS (site items): each active one needs its next step taken or its wait named:\n" + r.stdout)
    except Exception as e:                      # noqa: BLE001
        out.append("site items failed (%r): run it now." % e)
    return "\n".join(out)


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
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": AFTER + "\\n\\n" + plan_text()}}))
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
