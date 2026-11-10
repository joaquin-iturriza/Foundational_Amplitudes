#!/bin/bash
#SBATCH --job-name=ext_sig_eeuu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=analysis/divergences/ext_sig_eeuu_%j.out
#SBATCH --error=analysis/divergences/ext_sig_eeuu_%j.out
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd $PROJECT_DIR
python analysis/divergences/extract_train_pool_sigma.py --process eeuu
echo "DONE extract_sigma_eeuu"
