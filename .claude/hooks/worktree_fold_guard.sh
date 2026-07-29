#!/usr/bin/env bash
# PreToolUse(Bash) hook — refuse to delete a worktree whose RESULTS have not been folded
# back into the trunk.
#
# Why: `git merge` brings back a worktree's CODE and nothing else. Sweep results, run dirs,
# eval .npz, satlogs, generated datasets and figures are gitignored, so they exist only
# inside the worktree and are destroyed by `git worktree remove` / `rm -rf worktrees/...`.
# This has already lost real results: the 12-trial `sweeps/l2_poly_uugg` DyHPO sweep behind
# the polynomial-keep-rule figure no longer exists anywhere on disk, so that figure's left
# panel renders empty and cannot be rebuilt without re-running the sweep on GPU.
#
# The hook blocks the deletion and points at scripts/fold_worktree.sh, which copies the
# result files into the trunk. Once that script has run with --apply it records the
# worktree name, and the deletion is allowed through.
set -uo pipefail
REPO="/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"
FOLDED="$REPO/.claude/.review_state/folded_worktrees"

input=$(cat)
cmd=$(printf '%s' "$input" | python3 -c 'import sys,json
try:
    print(json.load(sys.stdin).get("tool_input",{}).get("command",""))
except Exception:
    print("")' 2>/dev/null)
[ -z "$cmd" ] && exit 0

# Only look at commands that destroy a worktree.
case "$cmd" in
  *"worktree remove"*|*"worktree  remove"*) ;;
  *rm\ *-*r*\ *worktrees/*|*rm\ -rf\ *worktrees/*) ;;
  *) exit 0 ;;
esac

# Which worktree(s)? Pull every worktrees/<name> token out of the command.
names=$(printf '%s' "$cmd" | grep -oE 'worktrees/[A-Za-z0-9._-]+' | sed 's#worktrees/##' | sort -u)
[ -z "$names" ] && exit 0

for n in $names; do
  wt="$REPO/worktrees/$n"
  [ -d "$wt" ] || continue
  # Self-validating check: is any result file in the worktree ABSENT from the trunk?
  # A name-keyed "already folded" record goes stale the moment a worktree of the same name
  # is recreated, and it cannot notice a fold that was incomplete -- which is exactly how
  # 327 config.yaml files were nearly lost. Ask the filesystem instead of a memo.
  pending=""
  while IFS= read -r src; do
    rel="${src#"$wt"/}"
    # tracked files come back through the merge; only untracked results are at risk
    if git -C "$wt" ls-files --error-unmatch "$rel" >/dev/null 2>&1; then continue; fi
    if [ ! -e "$REPO/$rel" ]; then pending="$rel"; break; fi
  done < <(find "$wt"/sweeps "$wt"/runs "$wt"/analysis "$wt"/plots "$wt"/compare_models \
                "$wt"/data_* "$wt"/satlogs -type f \
                \( -name '*.json' -o -name '*.yaml' -o -name '*.yml' -o -name '*.npz' \
                   -o -name '*.npy' -o -name '*.png' -o -name '*.pdf' -o -name '*.csv' \
                   -o -name '*.pkl' \) 2>/dev/null)
  [ -z "$pending" ] && continue

  {
    echo "BLOCKED by worktree_fold_guard: worktree 'worktrees/$n' still holds results that"
    echo "exist NOWHERE ELSE. Merging the branch does not move them -- they are gitignored,"
    echo "so removing the worktree destroys them permanently."
    echo
    echo "Fold them into the trunk first:"
    echo "  bash scripts/fold_worktree.sh worktrees/$n              # see what would be copied"
    echo "  bash scripts/fold_worktree.sh worktrees/$n --apply      # copy them"
    echo
    echo "First still-missing file: $pending"
    echo "Then retry the removal. (This already cost the l2_poly_uugg sweep once.)"
  } >&2
  exit 2
done
exit 0
