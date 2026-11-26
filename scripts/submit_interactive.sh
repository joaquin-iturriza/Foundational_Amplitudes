#!/bin/bash
# Submitter only: starts a job script on the gpu_v100_interactive partition (srun in the background), runs nothing itself.
#   bash scripts/submit_interactive.sh <job script> <log path> [VAR=VALUE ...]
set -eu
job=$1; log=$2; shift 2
for kv in "$@"; do export "$kv"; done
export CCORCH_SITE=ccin2p3 CCORCH_PROJECT_DIR=$PWD
nohup setsid srun --partition=gpu_v100_interactive --qos=gpu --account=lpnhe --gpus=1 --mem=32G --cpus-per-task=5 \
    -t 02:00:00 -J "int_$(basename "$job" .sh)" bash "$job" > "$log" 2>&1 < /dev/null &
echo "$! submitted"
