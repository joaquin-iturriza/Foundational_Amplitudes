#!/bin/bash
#SBATCH --job-name=het_rank
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=analysis/divergences/het_rank_%j.out
#SBATCH --error=analysis/divergences/het_rank_%j.out
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd $PROJECT_DIR
python analysis/divergences/rank_mu_mse.py --mse_ref ""
echo "DONE eval_rank"
