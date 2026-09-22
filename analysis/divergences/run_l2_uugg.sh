#!/bin/bash
#SBATCH --job-name=l2uugg
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=analysis/divergences/l2uugg_%x_%j.out
#SBATCH --error=analysis/divergences/l2uugg_%x_%j.out
#SBATCH --gres=gpu:1
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"/worktrees/wt-l2-uugg
export PYTHONDONTWRITEBYTECODE=1
ARM=${ARM:-sigma}; SEED=${SEED:-0}; TAG=${TAG:-run}
python analysis/divergences/l2_online_uugg.py \
  --arm $ARM --tag $TAG --total_steps 4000 --n_total 300000 --rounds 10 \
  --oversample ${OVERSAMPLE:-4} --gamma ${GAMMA:-1.0} --y_lo 1e-6 --mix_ir 0.5 --sigma0 0.1 --seed $SEED \
  --heldout_eval
echo "DONE l2uugg arm=$ARM seed=$SEED"
