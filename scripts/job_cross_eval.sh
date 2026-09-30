#!/bin/bash
#SBATCH --job-name=cross_eval
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# Score finished runs on another recipe's split (analysis/transfer/cross_eval.py).
#     site submit <site> FA scripts/job_cross_eval.sh -- recipes/<recipe>.yaml <run_dir> [<run_dir> ...]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
R="$1"; shift
python analysis/transfer/cross_eval.py --recipe "$R" "$@"          # --role test by default; --dump saves residuals
