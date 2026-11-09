#!/usr/bin/env bash
# Stop hook — push any unpushed commits to origin automatically at end of turn.
#
# Why: the git workflow says "push after every significant change", but relying on
# the model to remember (and not ask permission) is unreliable. This makes it
# deterministic. It only pushes work that is ALREADY COMMITTED — it never creates
# commits and never touches `main` (a generated artifact). Failures are non-fatal
# (printed, exit 0) so a push problem never blocks the session; trunk goes out only up to
# its last claims-verified commit.
set -uo pipefail
REPO="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO" 2>/dev/null || exit 0

# The numbers in docs/results.tex are checked as state (setup redesign, D121): trunk is pushed only up to the last
# commit the claims check verified (`plan claims --verified` reads it, nothing runs; plan_check.py runs the check and
# blocks the stop on a problem). Newer commits wait for it, so the two Stop hooks never race on the check's cache.
gated=""
verified=""
if [ -f "$REPO/plan/plan.yaml" ] && [ -f "$REPO/tools/plan.py" ]; then
  gated=1
  verified=$(python3 "$REPO/tools/plan.py" claims --verified 2>/dev/null)
fi

# Walk the worktrees for the one on trunk (the main checkout); every other branch is skipped.
git worktree list --porcelain 2>/dev/null | awk '/^worktree /{print $2}' | while read -r wt; do
  br=$(git -C "$wt" symbolic-ref --quiet --short HEAD 2>/dev/null) || continue
  [ -z "$br" ] && continue
  [ "$br" = "trunk" ] || continue   # main is published by scripts/publish_main.sh; feature branches stay local
  if [ "$br" = "trunk" ] && [ -n "$gated" ]; then
    head=$(git -C "$wt" rev-parse HEAD)
    if [ "$verified" != "$head" ]; then
      if [ -n "$verified" ] && git -C "$wt" merge-base --is-ancestor "$verified" "$head" 2>/dev/null; then
        new=$(git -C "$wt" rev-list --count '@{u}'.."$verified" 2>/dev/null || echo 1)
        wait=$(git -C "$wt" rev-list --count "$verified".."$head")
        if [ "${new:-0}" -gt 0 ]; then
          if git -C "$wt" push --quiet origin "$verified:refs/heads/trunk" 2>/dev/null; then
            echo "[auto-push] trunk: pushed up to the last verified commit ${verified:0:8}; $wait newer wait for the claims check"
          else
            echo "[auto-push] trunk: push up to ${verified:0:8} FAILED (push manually)"
          fi
        else
          echo "[auto-push] trunk: $wait commit(s) wait for the claims check (plan_check.py, or plan claims)"
        fi
      else
        echo "[auto-push] trunk: not pushed; no verified commit on it (run python3 tools/plan.py claims)"
      fi
      continue
    fi
  fi

  # Only trunk goes to origin (D334, 2026-11-09: 97 remote branches, most merged feature, agent and trial branches
  # this hook had published and nothing deleted). A feature branch lives in its worktree until it merges into trunk,
  # which is what reaches origin; scripts/remove_worktree.sh deletes any remote copy left from before.
  ahead=$(git -C "$wt" rev-list --count '@{u}'..HEAD 2>/dev/null || echo 1)
  if [ "${ahead:-0}" -gt 0 ]; then
    if git -C "$wt" push --quiet origin "HEAD:refs/heads/trunk" 2>/dev/null; then
      echo "[auto-push] trunk: pushed $ahead commit(s) to origin"
    else
      echo "[auto-push] trunk: push of $ahead commit(s) FAILED (push manually)"
    fi
  fi
done
exit 0

