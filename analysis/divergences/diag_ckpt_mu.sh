#!/bin/bash
#SBATCH --job-name=diag_mu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=analysis/divergences/diag_mu_%j.out
#SBATCH --error=analysis/divergences/diag_mu_%j.out
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd $PROJECT_DIR
python analysis/divergences/diag_ckpt_mu.py
echo "DONE diag_ckpt_mu"
