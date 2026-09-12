#!/usr/bin/env bash
# remote.sh — run a job-control command in the Foundational_Amplitudes project
# dir on CC-IN2P3.
#
# Same model as madgrav/Fin_ML on this cluster: the assistant runs LOCALLY
# against an sshfs mount of the project, so local
# /home/joaquin/mnt/ccin2p3/Foundational_Amplitudes IS remote
# /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes (same bytes). All file
# work — read, search, edit, tail logs — happens on the mount with no ssh at
# all. Only commands that genuinely need the scheduler cross the wire.
#
# Usage:
#   scripts/remote.sh sbatch --parsable jobs/job_train.sh
#   scripts/remote.sh squeue --me
#   scripts/remote.sh sacct -j <id> --format=JobID,State,ExitCode,Elapsed
#
# The command runs after `cd <project>` on the login node, so relative paths
# (jobs/..., runs/...) resolve exactly as a manual login-node submit would.
#
# Env overrides: CCIN2P3_HOST (ssh alias, default ccin2p3),
#                CCIN2P3_PROJ (remote project dir).
set -euo pipefail
HOST="${CCIN2P3_HOST:-ccin2p3}"
PROJ="${CCIN2P3_PROJ:-/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes}"

if [ "$#" -eq 0 ]; then
  echo "usage: scripts/remote.sh <command to run in the project dir on the cluster>" >&2
  exit 2
fi

# The login shell on the cluster is where the module system and any env live;
# a bare `ssh host cmd` gets a non-login shell with a minimal PATH.
# BatchMode: a hook or background waiter must fail, not sit on a password prompt.
exec ssh -o BatchMode=yes -o ConnectTimeout="${CCIN2P3_CONNECT_TIMEOUT:-20}" "$HOST" \
  "cd '$PROJ' && bash -lc $(printf '%q' "$*")"
