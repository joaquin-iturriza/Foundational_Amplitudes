#!/usr/bin/env bash
# Fold a worktree's RESULTS back into the trunk before the worktree is deleted.
#
# WHY THIS EXISTS. Code in a worktree comes back through `git merge`. Everything else
# does not: sweep results, run dirs, eval .npz, satlogs, generated datasets and figures
# are all gitignored, so they live only in the worktree's own directory tree and die with
# `git worktree remove`. That has already cost real results here -- the 12-trial
# `sweeps/l2_poly_uugg` DyHPO sweep behind the polynomial-keep-rule figure is gone, so the
# left panel of that figure now renders empty, and several plotting scripts reference
# eval files that no longer exist.
#
# Merging the branch is NOT enough. Run this before removing any worktree.
#
# Usage:
#   scripts/fold_worktree.sh worktrees/wt-foo            # dry run: list what would be copied
#   scripts/fold_worktree.sh worktrees/wt-foo --apply    # copy it into the trunk
#
# It copies only files that are MISSING from the trunk, or whose worktree mtime is newer
# than the trunk copy's. It never deletes and never overwrites a trunk file with an older
# or identical one, so running it twice is safe.
set -uo pipefail

REPO="/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes"

WT="${1:-}"
APPLY=0
[ "${2:-}" = "--apply" ] && APPLY=1
if [ -z "$WT" ]; then
  echo "usage: $0 <worktree-path> [--apply]" >&2
  exit 2
fi
[ -d "$WT" ] || { echo "no such worktree: $WT" >&2; exit 2; }
WT=$(cd "$WT" && pwd)
[ "$WT" = "$REPO" ] && { echo "refusing to fold the trunk into itself" >&2; exit 2; }

# Directories that carry results rather than source. These are exactly the things
# gitignored (so a merge does not move them) but expensive or impossible to regenerate.
RESULT_DIRS="sweeps runs analysis plots data logs compare_models"
# Extensions worth folding when they turn up inside those dirs.
# yaml is NOT optional: a run's config.yaml is the only record of what it trained on
# (data.processes_file, lr, preprocessing). Folding the numbers without it produces results
# that look citable but have no provenance -- the exact failure this script exists to stop.
# sh covers generated sweep job scripts (jobs/trial_*.sh).
RESULT_EXTS="json yaml yml npz npy png pdf csv pkl txt log out sh"

echo "folding results:  $WT"
echo "            into: $REPO"
[ "$APPLY" = 0 ] && echo "(DRY RUN -- pass --apply to copy)"
echo

# Tracked files come back through `git merge`; they are NOT at risk and must not be
# copied (their worktree mtime is just the checkout time, which would look "newer" and
# could clobber a trunk file with staler content). Only untracked/ignored files are lost.
tracked=$(mktemp)
git -C "$WT" ls-files > "$tracked" 2>/dev/null

n_new=0; n_newer=0; n_same=0; n_tracked=0; bytes=0
tmp=$(mktemp)
for d in $RESULT_DIRS; do
  [ -d "$WT/$d" ] || continue
  find "$WT/$d" -type f 2>/dev/null >> "$tmp"
done
# also fold any satlogs / data_* dirs sitting at the worktree root
for extra in "$WT"/data_* "$WT"/satlogs; do
  [ -d "$extra" ] && find "$extra" -type f 2>/dev/null >> "$tmp"
done

while IFS= read -r src; do
  [ -f "$src" ] || continue
  ext="${src##*.}"
  case " $RESULT_EXTS " in *" $ext "*) ;; *) continue ;; esac
  rel="${src#"$WT"/}"
  if grep -qxF "$rel" "$tracked" 2>/dev/null; then
    n_tracked=$((n_tracked+1)); continue
  fi
  dst="$REPO/$rel"
  if [ ! -e "$dst" ]; then
    n_new=$((n_new+1))
    sz=$(stat -c%s "$src" 2>/dev/null || echo 0); bytes=$((bytes+sz))
    [ "$n_new" -le 20 ] && echo "  NEW    $rel"
    if [ "$APPLY" = 1 ]; then mkdir -p "$(dirname "$dst")" && cp -p "$src" "$dst"; fi
  elif [ "$src" -nt "$dst" ]; then
    n_newer=$((n_newer+1))
    [ "$n_newer" -le 10 ] && echo "  NEWER  $rel"
    if [ "$APPLY" = 1 ]; then cp -p "$src" "$dst"; fi
  else
    n_same=$((n_same+1))
  fi
done < "$tmp"
rm -f "$tmp" "$tracked"

echo
echo "  new in worktree      : $n_new  (~$((bytes/1024)) KiB)"
echo "  newer than trunk     : $n_newer"
echo "  already in trunk     : $n_same"
if [ "$APPLY" = 1 ]; then
  echo
  echo "COPIED. Commit anything that belongs in git, then the worktree is safe to remove:"
  echo "  git worktree remove $WT"
  # No "folded" record is written on purpose: worktree_fold_guard.sh re-checks the
  # filesystem instead, so an incomplete fold cannot be masked by a memo saying it is done.
elif [ $((n_new + n_newer)) -gt 0 ]; then
  echo
  echo "Nothing copied yet. Re-run with --apply before removing this worktree."
fi
