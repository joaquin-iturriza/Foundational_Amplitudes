#!/bin/bash
#SBATCH --job-name=ev_flatlogm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=analysis/divergences/ev_flatlogm_%j.out
#SBATCH --error=analysis/divergences/ev_flatlogm_%j.out

set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd $PROJECT_DIR

# eval script lives on the heterosc branch (worktree only); run configs + ckpts are in MAIN.
# Three-way: raw RAMBO baseline vs flatlogm (RESAMPLED fixed pool -> coverage ceiling, fails)
# vs genflat (GENERATED fresh flat-log|M|^2 events -> breaks the ceiling). Each de-standardized
# with its OWN per-pool stats back to true log|M|^2 (comparable), binned by sqrt(s).
python analysis/divergences/eval_eeuu_resonance.py \
  --runs_root runs/eeuu_flatlogm --run_prefix ft_ --tags raw,flatlogm,genflat

echo "DONE eval_flatlogm"
