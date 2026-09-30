#!/bin/bash
#SBATCH --job-name=steer_pool
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# Select a sigma-steered pool (tools/steer_pool.py --stage select): score the candidates with the
# recipe's reference model, keep prop. to sigma^gamma, write train/val/test. The candidates are
# labelled first on CPU (scripts/job_steer_candidates.sh), so this GPU job generates nothing.
#     site submit <site> FA scripts/job_steer_pool.sh -- recipes/<steered recipe>.yaml
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
python tools/steer_pool.py --recipe "$1" --stage select
