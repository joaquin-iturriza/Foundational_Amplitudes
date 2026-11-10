#!/bin/bash
#SBATCH --job-name=ev_genmix
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=analysis/divergences/ev_genmix_%j.out
#SBATCH --error=analysis/divergences/ev_genmix_%j.out

set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd $PROJECT_DIR

# Full generated fraction sweep: f = 0 (raw) .. 1 (genflat). Each de-standardized to true
# log|M|^2, common RAMBO test binned by sqrt(s). Locates the interior optimum.
python analysis/divergences/eval_eeuu_resonance.py \
  --runs_root runs/eeuu_flatlogm --run_prefix ft_ \
  --tags raw,mix025,mix050,mix075,genflat \
  --out_prefix eeuu_reson_sw_ --summary eeuu_reson_sweep_summary.json

echo "DONE eval_genmix"
