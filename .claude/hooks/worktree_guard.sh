#!/usr/bin/env bash
# PreToolUse(Edit|Write|MultiEdit) hook — worktree reminder.
#
# Why: "create a worktree for new feature work" is a proactive step with no
# natural trigger, so the model forgets it. This supplies the trigger at the exact
# moment it matters: the first edit of trunk code on `jeanzay`. It is ADVISORY
# (non-blocking) — it injects a reminder and lets the edit proceed, so false
# positives on quick/standalone edits cost one line, not a hard stop.
set -uo pipefail
REPO="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

input=$(cat)
fp=$(printf '%s' "$input" | python3 -c 'import sys,json;
try:
    print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))
except Exception:
    print("")' 2>/dev/null)
[ -z "$fp" ] && exit 0

# Only files in this repo matter.
case "$fp" in
  "$REPO"/*) ;;
  *) exit 0 ;;
esac

# Feature worktrees live UNDER the repo at worktrees/wt-<feat> — editing inside one
# is already "in a worktree", so never nudge there.
case "$fp" in
  "$REPO"/worktrees/*) exit 0 ;;
esac

# Exempt meta / lightweight files that don't warrant a feature worktree.
case "$fp" in
  */.claude/*|*/CLAUDE.md|*.md|*/recipes/*|*/scratchpad/*) exit 0 ;;
esac

# Only nudge when actually sitting on the trunk branch in the main checkout.
br=$(git -C "$REPO" symbolic-ref --quiet --short HEAD 2>/dev/null)
case "$br" in jeanzay|ccin2p3) ;; *) exit 0 ;; esac

rel=${fp#"$REPO"/}
msg="Worktree reminder: editing trunk file '$rel' directly on $br. Per the git workflow, new feature work should go in a worktree created UNDER this repo (git worktree add worktrees/wt-<feat> -b <feat> $br), never outside it (no ../wt-<feat>). If this is a quick standalone edit, proceed; otherwise create a worktree first."
# Inject as additional context to the model without blocking the edit.
printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":%s}}\n' \
  "$(printf '%s' "$msg" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')"
exit 0
