#!/bin/bash
#SBATCH --job-name=pole_width_diag
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=pole_width_diag_%j.out
#SBATCH --error=pole_width_diag_%j.out
#
# CPU only: the double-pole check of catalog_v3's 68 b/t one-loop strings with the top width zeroed and as generated
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
python tools/pole_width_diag.py --stage2-bt --n 200
