#!/bin/bash
#SBATCH --job-name=smoke_l2uugg
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=analysis/divergences/smoke_l2uugg_%j.out
#SBATCH --error=analysis/divergences/smoke_l2uugg_%j.out
#SBATCH --gres=gpu:1

set -e
module purge
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"/worktrees/wt-l2-uugg
export PYTHONDONTWRITEBYTECODE=1

# tiny end-to-end smoke: 40 steps, 4 rounds (10 steps/round), 2000 events (500/round).
# Exercises warm-start base22, the _online_hook firing at rounds 1/2/3 (generate->label->extend),
# the CONTINUOUS cosine, and final eval. Cost ~1-2 GPU-min.
python analysis/divergences/l2_online_uugg.py \
  --arm base --tag smoke --total_steps 12 --rounds 3 --n_total 1500 \
  --oversample 4 --y_lo 1e-6 --mix_ir 0.5 --seed 0

echo "DONE smoke"
