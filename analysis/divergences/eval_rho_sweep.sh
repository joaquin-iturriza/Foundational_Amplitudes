#!/bin/bash
#SBATCH --job-name=q2rhoev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=analysis/divergences/q2rhoev_%j.out
#SBATCH --error=analysis/divergences/q2rhoev_%j.out
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd $PROJECT_DIR
python analysis/divergences/eval_heldout.py \
  --runs_root runs/q2_sigma_reweight \
  --run_prefix ft_ --tags rho10,rho20,rho30,rho40,rho50,rho60,rho70,rho80,rho90 \
  --heldout analysis/divergences/uug_deep_test.npz \
  --out_prefix q2rw_eval_ \
  --summary q2rho_eval_summary.json
echo "DONE q2 rho eval"
