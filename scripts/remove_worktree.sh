#!/bin/bash
# Remove a finished worktree the one safe way: fold its untracked results into the trunk, then remove it (--force only
# after every untracked file is checked), then delete its branch if it is merged. The steps cannot be skipped or reordered by the caller.
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
# regenerable caches (__pycache__/ dirs, *.pyc, .pytest_cache/) never count: they are never a result
left=()
while IFS= read -r f; do
    case "$f" in */__pycache__/*|__pycache__/*|*.pyc|.pytest_cache/*|*/.pytest_cache/*) continue ;; esac
    cmp -s "$WT/$f" "$REPO/$f" || left+=("$f")
done < <(git -C "$WT" ls-files --others; git -C "$WT" ls-files --others --ignored --exclude-standard)
if [ ${#left[@]} -gt 0 ]; then
    echo "remove_worktree: $WT keeps ${#left[@]} untracked file(s) with no identical copy in the trunk; not removed:" >&2
    printf '  %s\n' "${left[@]}" | head -20 >&2
    exit 1
fi
git worktree remove --force "$WT"
if git branch --merged trunk | grep -qx "  $BR"; then
    git branch -d "$BR"
else
    echo "remove_worktree: branch $BR is not merged into trunk; kept"
fi
