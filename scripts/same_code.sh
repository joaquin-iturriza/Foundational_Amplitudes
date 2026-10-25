#!/bin/bash
# Exit 0 when two commits carry the same job code: they differ only in plan/, the plan file every plan write commits
# to trunk, which no job reads. A site checkout a few plan
# commits behind runs exactly the code this checkout has, so a NOSYNC launch against it is right (2026-10-25:
# with every plan write a commit, the launcher's equality check lost its race again and again).
#     bash scripts/same_code.sh <site commit> [<local commit, default HEAD>]
cd "$(dirname "$0")/.." || exit 2
a="$1"; b="${2:-HEAD}"
[ -n "$a" ] || exit 2
git cat-file -e "$a^{commit}" 2>/dev/null || exit 1      # a commit this checkout does not know: not shown the same
git cat-file -e "$b^{commit}" 2>/dev/null || exit 1
git diff --quiet "$a" "$b" -- . ':(exclude)plan'
