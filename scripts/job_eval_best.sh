#!/bin/bash
#SBATCH --job-name=eval_best
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# Score runs' best checkpoints on their own validation split (tools/eval_best_val.py).
#     site submit <site> FA scripts/job_eval_best.sh -- <run_dir> [<run_dir> ...]
# LIVE_TOO=1 scores each run a second time on the checkpoint's "model" entry alone (EVAL_LIVE=1: the live weights of a
# pre-D63 EMA checkpoint, what the tool scored before 2026-10-15)
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
python tools/eval_best_val.py "$@"
[ "${LIVE_TOO:-0}" = 1 ] && EVAL_LIVE=1 python tools/eval_best_val.py "$@"
