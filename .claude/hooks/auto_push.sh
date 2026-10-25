#!/usr/bin/env bash
# Stop hook — push any unpushed commits to origin automatically at end of turn.
#
# Why: the git workflow says "push after every significant change", but relying on
# the model to remember (and not ask permission) is unreliable. This makes it
# deterministic. It only pushes work that is ALREADY COMMITTED — it never creates
# commits and never touches `main` (a generated artifact). Push failures are non-fatal
# (printed, exit 0); only a failed claims check holds trunk back and blocks the stop (exit 2).
set -uo pipefail
REPO="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO" 2>/dev/null || exit 0

# The numbers in docs/results.tex are checked as state (setup redesign, D121): trunk is not pushed while a commit
# since the last verified one fails tools/verify_claims.py (`plan claims`); the stop is blocked with the problems.
hold_trunk=""
if [ -f "$REPO/plan/plan.yaml" ] && [ -f "$REPO/tools/plan.py" ]; then
  if ! claims=$(python3 "$REPO/tools/plan.py" claims 2>&1); then
    hold_trunk=1
    printf '[auto-push] trunk NOT pushed: the claims check fails\n%s\n' "$claims" >&2
  fi
fi

# Walk every worktree (the main checkout + any ../wt-* feature worktrees) so
# feature-branch commits get pushed too.
git worktree list --porcelain 2>/dev/null | awk '/^worktree /{print $2}' | while read -r wt; do
  br=$(git -C "$wt" symbolic-ref --quiet --short HEAD 2>/dev/null) || continue
  [ -z "$br" ] && continue
  [ "$br" = "main" ] && continue   # main is published via scripts/publish_main.sh, never pushed here
  [ "$br" = "trunk" ] && [ -n "$hold_trunk" ] && continue

  if git -C "$wt" rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    ahead=$(git -C "$wt" rev-list --count '@{u}'..HEAD 2>/dev/null || echo 0)
    if [ "${ahead:-0}" -gt 0 ]; then
      if git -C "$wt" push --quiet 2>/dev/null; then
        echo "[auto-push] $br: pushed $ahead commit(s) to origin"
      else
        echo "[auto-push] $br: push of $ahead commit(s) FAILED (push manually)"
      fi
    fi
  else
    # New feature branch with no upstream yet — publish it.
    if git -C "$wt" push --quiet -u origin "$br" 2>/dev/null; then
      echo "[auto-push] $br: published new branch to origin"
    else
      echo "[auto-push] $br: publish FAILED (push manually)"
    fi
  fi
done
[ -n "$hold_trunk" ] && exit 2
exit 0
