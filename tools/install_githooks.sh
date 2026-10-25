#!/bin/sh
# Point this repository (and every worktree, which share its config) at the tracked hooks in .githooks: the claims
# check on any commit or merge that changes docs/results.tex. Run by tools/switch_over.sh (setup redesign, D121).
set -e
top=$(git rev-parse --show-toplevel)
chmod +x "$top/.githooks/"*
git -C "$top" config core.hooksPath .githooks
echo "core.hooksPath = $(git -C "$top" config core.hooksPath)"
