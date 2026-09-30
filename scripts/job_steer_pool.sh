#!/bin/bash
#SBATCH --job-name=steer_pool
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# Build a sigma-steered pool once (tools/steer_pool.py): label the candidates, score them with the
# recipe's reference model, keep prop. to sigma^gamma, write train/val/test.
#     site submit <site> FA scripts/job_steer_pool.sh -- recipes/<steered recipe>.yaml
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
python tools/steer_pool.py --recipe "$1"
