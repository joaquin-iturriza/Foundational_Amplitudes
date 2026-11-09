#!/bin/bash
# Remove a finished worktree the one safe way: fold its untracked results into the trunk, then remove it (--force only
# after every untracked file is checked), then delete its branch if it is merged, here and on origin. The steps cannot be skipped or reordered by the caller.
#     bash scripts/remove_worktree.sh worktrees/wt-<feat> [--discard]
# A detached worktree (a model-trial arm) has no branch to keep its commits: it is removed only when its HEAD is in
# trunk, or with --discard, which drops its commits (their sha is printed, recoverable while git keeps them).
set -euo pipefail
WT="${1:?usage: remove_worktree.sh worktrees/wt-<feat> [--discard]}"
DISCARD=0
[ "${2:-}" = "--discard" ] && DISCARD=1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[ -d "$WT" ] || { echo "remove_worktree: no worktree at $WT" >&2; exit 1; }
BR="$(git -C "$WT" rev-parse --abbrev-ref HEAD)"
SHA="$(git -C "$WT" rev-parse HEAD)"
if [ "$BR" = "HEAD" ] && ! git merge-base --is-ancestor "$SHA" trunk; then
    if [ $DISCARD -eq 0 ]; then
        echo "remove_worktree: $WT is detached at $SHA, which trunk lacks; not removed (--discard drops it)" >&2; exit 1
    fi
    echo "remove_worktree: discarding detached $WT at $SHA (not in trunk)"
fi
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
    # the sha is read from origin itself (a fetch that failed would leave a stale tracking ref), must be a commit here
    # and in trunk, and the delete is leased on it: a push to origin/$BR after the read makes the delete fail
    if ! heads="$(git ls-remote --heads origin "refs/heads/$BR" 2>/dev/null)"; then
        echo "remove_worktree: origin unreachable; origin/$BR not checked"
        exit 0
    fi
    rsha="$(printf '%s\n' "$heads" | awk -v r="refs/heads/$BR" '$2 == r {print $1}')"
    if [ -n "$rsha" ]; then
        if git cat-file -e "$rsha^{commit}" 2>/dev/null && git merge-base --is-ancestor "$rsha" trunk; then
            git push --quiet --force-with-lease="refs/heads/$BR:$rsha" origin --delete "$BR" &&
                echo "remove_worktree: deleted origin/$BR"
        else
            echo "remove_worktree: origin/$BR ($rsha) has commits trunk lacks; kept"
        fi
    fi
else
    echo "remove_worktree: branch $BR is not merged into trunk; kept"
fi
