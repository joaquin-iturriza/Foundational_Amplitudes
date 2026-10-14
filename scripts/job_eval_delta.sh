#!/bin/bash
#SBATCH --job-name=eval_delta
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# Per-event test/val predictions of runs' best checkpoints for the Delta plots (tools/eval_delta.py, D42).
#     site submit <site> FA scripts/job_eval_delta.sh -- <run_dir> [<run_dir> ...]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
python tools/eval_delta.py --out-dir "$SCRATCH/zz_delta" "$@"
