#!/bin/bash
# Remove a finished worktree the one safe way: fold its untracked results into the trunk, then remove it (--force only
# after every untracked file is checked), then delete its branch if it is merged, here and on origin. The steps cannot be skipped or reordered by the caller.
#     bash scripts/remove_worktree.sh worktrees/wt-<feat>
set -euo pipefail
WT="${1:?usage: remove_worktree.sh worktrees/wt-<feat>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[ -d "$WT" ] || { echo "remove_worktree: no worktree at $WT" >&2; exit 1; }
BR="$(git -C "$WT" rev-parse --abbrev-ref HEAD)"
if [ -n "$(git -C "$WT" status --porcelain --untracked-files=no)" ]; then
    echo "remove_worktree: $WT has uncommitted tracked changes; commit or drop them first" >&2; exit 1
fi
bash scripts/fold_worktree.sh "$WT" --apply
# git refuses to remove a worktree holding untracked files; --force is used only once every untracked file has an
# identical copy at the same path in the trunk (the fold put the results there); anything else is listed and kept
# regenerable caches (__pycache__/ dirs, *.pyc, .pytest_cache/) and per-checkout runtime state never count
left=()
while IFS= read -r f; do
    # per-checkout runtime state (review watermarks, watchdog state, the plan's check cache and state) is bookkeeping of
    # that checkout, never a result: the trunk holds its own (D334, 13 merged worktrees refused on these alone)
    case "$f" in */__pycache__/*|__pycache__/*|*.pyc|.pytest_cache/*|*/.pytest_cache/*) continue ;;
        .claude/.review_state/*|.claude/.watchdog_state.json|.claude/.notes_flushed|plan/.check_cache.json|plan/.state.json) continue ;;
        docs/*.aux|docs/*.log|docs/*.out|docs/*.toc) continue ;; esac     # LaTeX build products of docs/*.tex
    cmp -s "$WT/$f" "$REPO/$f" || left+=("$f")
done < <(git -C "$WT" ls-files --others; git -C "$WT" ls-files --others --ignored --exclude-standard)
if [ ${#left[@]} -gt 0 ]; then
    echo "remove_worktree: $WT keeps ${#left[@]} untracked file(s) with no identical copy in the trunk; not removed:" >&2
    printf '  %s\n' "${left[@]}" | head -20 >&2
    exit 1
fi
git worktree remove --force "$WT"
if [ "$BR" = "HEAD" ] || [ "$BR" = "trunk" ] || [ "$BR" = "main" ]; then
    exit 0                       # a detached worktree (model trials) has no branch to delete
fi
if git merge-base --is-ancestor "$BR" trunk; then
    # merged: -D, since -d also asks for the branch to be merged into its upstream, and a stale upstream (an old
    # auto_push of the branch) made it refuse a merged branch (2026-11-04, double_wake)
    git branch -D "$BR"
    # its copy on origin goes too, when that copy is also in trunk (D334: no merged branches left on origin)
    if git ls-remote --exit-code --heads origin "$BR" >/dev/null 2>&1; then
        git fetch --quiet origin "refs/heads/$BR:refs/remotes/origin/$BR" 2>/dev/null || true
        if git merge-base --is-ancestor "refs/remotes/origin/$BR" trunk 2>/dev/null; then
            git push --quiet origin --delete "$BR" && echo "remove_worktree: deleted origin/$BR"
        else
            echo "remove_worktree: origin/$BR has commits trunk lacks; kept"
        fi
    fi
else
    echo "remove_worktree: branch $BR is not merged into trunk; kept"
fi
