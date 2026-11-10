#!/bin/bash
#SBATCH --job-name=l2_online
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=analysis/divergences/l2_online_%x_%j.out
#SBATCH --error=analysis/divergences/l2_online_%x_%j.out
# Usage: sbatch --job-name=l2_<arm> l2_online.sh --arm <arm> [--rounds N --iters N ...]
# Runs the whole online L2 loop for ONE arm inside ONE GPU allocation (each GPU stage is its
# own fresh run.py/py subprocess to avoid the muP cross-instance global leak).
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd $PROJECT_DIR
python analysis/divergences/l2_online.py "$@"
echo "DONE l2_online $*"
