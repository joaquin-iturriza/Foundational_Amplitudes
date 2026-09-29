#!/bin/bash
#SBATCH --job-name=measure_id
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# Intrinsic dimension of trained models' best checkpoints (tools/measure_id.py); arguments pass through.
#     site submit <site> FA scripts/job_measure_id.sh -- --out analysis/id/<name>.jsonl --sweep <dir> [...]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
mkdir -p analysis/id
python tools/measure_id.py "$@"
