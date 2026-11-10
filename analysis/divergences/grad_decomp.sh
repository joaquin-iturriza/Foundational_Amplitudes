#!/bin/bash
#SBATCH --job-name=graddec
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=analysis/divergences/graddec_%j.out
#SBATCH --error=analysis/divergences/graddec_%j.out
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd $PROJECT_DIR
python analysis/divergences/grad_decomp.py
echo DONE
