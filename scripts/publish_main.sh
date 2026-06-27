#!/usr/bin/env bash
#
# publish_main.sh — regenerate the public `main` branch as a curated subset of
# the `jeanzay` development trunk, then push it.
#
# `main` is a BUILD ARTIFACT of `jeanzay`: never edit `main` by hand. To change
# what is public, edit the PUBLIC_PATHS allowlist below and re-run this script.
# Anything not in the allowlist is removed from `main`; everything in it is
# synced from `jeanzay`. History on `main` is preserved (each publish is a new
# commit on top).
#
# Usage:  scripts/publish_main.sh [--no-push]
#
set -euo pipefail

SRC="jeanzay"          # development trunk (source of truth)
DST="main"             # public branch (generated)
PUSH=1
[[ "${1:-}" == "--no-push" ]] && PUSH=0

# --- the public core: only these top-level paths are published ---------------
PUBLIC_PATHS=(
  .gitignore
  requirements.txt
  README.md
  # --- core run path ---
  run.py
  experiment.py
  base_experiment.py
  dataset.py
  wrappers.py
  preprocessing.py
  particle_ids.py
  misc.py
  losses.py
  plots.py
  base_plots.py
  mlflow_util.py
  logger.py
  fine_tune.py
  # --- on-the-fly data generation engine ---
  datagen.py
  mg5_pipeline_final.py
  prebuild_recipes.py
  # --- packages / configs / vendored deps / examples ---
  config
  models
  IntrinsicDimDeep
  recipes
)

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

# `main` must exist locally; seed it from origin/main (or from SRC) the first time.
if ! git show-ref --verify --quiet "refs/heads/${DST}"; then
  if git show-ref --verify --quiet "refs/remotes/origin/${DST}"; then
    git branch "${DST}" "origin/${DST}"
  else
    git branch "${DST}" "${SRC}"
  fi
fi

wt="$(mktemp -d)"
cleanup() { git worktree remove --force "$wt" >/dev/null 2>&1 || true; }
trap cleanup EXIT

git worktree add --force "$wt" "${DST}" >/dev/null

(
  cd "$wt"

  # 1. remove any tracked top-level entry that is not in the allowlist
  while IFS= read -r entry; do
    keep=0
    for p in "${PUBLIC_PATHS[@]}"; do [[ "$entry" == "$p" ]] && keep=1 && break; done
    [[ $keep -eq 0 ]] && git rm -rq "$entry"
  done < <(git ls-tree --name-only HEAD)

  # 2. sync every allowlisted path from the trunk (picks up new files + updates)
  for p in "${PUBLIC_PATHS[@]}"; do
    git checkout "${SRC}" -- "$p" 2>/dev/null || echo "warn: '$p' not found on ${SRC}, skipping"
  done

  git add -A
  if git diff --cached --quiet; then
    echo "main is already in sync with the public subset of ${SRC} — nothing to publish."
    exit 0
  fi

  git commit -q -m "Publish: sync public core from ${SRC}@$(git rev-parse --short "${SRC}")"
  echo "Published $(git rev-parse --short HEAD) on ${DST}."
  if [[ $PUSH -eq 1 ]]; then
    git push origin "${DST}"
  else
    echo "(--no-push: review, then 'git push origin ${DST}')"
  fi
)
