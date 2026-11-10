#!/bin/bash
#SBATCH --job-name=q2ev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=analysis/divergences/q2ev_%j.out
#SBATCH --error=analysis/divergences/q2ev_%j.out
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd $PROJECT_DIR

# Forward the COMMON deep-IR held-out test through each reweighting arm, recovering each
# run's own preprocessing stats -> deep-IR-binned MSE(log|M|^2) on one absolute scale.
python analysis/divergences/eval_heldout.py \
  --runs_root runs/q2_sigma_reweight \
  --run_prefix ft_ --tags baseQ,oracle,sigma,deg029 \
  --heldout analysis/divergences/uug_deep_test.npz \
  --out_prefix q2rw_eval_ \
  --summary q2rw_eval_summary.json
echo "DONE q2 eval"
