#!/bin/bash
#SBATCH --job-name=prebuild
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# CPU-only: materialize recipe pools in this site's cache (prebuild_recipes.py, as sweep_manager's auto-emitted
# prebuild does on SLURM), for sites whose sweep path has no prebuild step (HTCondor DAGs).
#     site submit <site> FA scripts/job_prebuild.sh -- recipes/<a>.yaml [recipes/<b>.yaml ...]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
set -euo pipefail
for spec in "$@"; do
  python prebuild_recipes.py "$spec" --seed 42 --workers "${SLURM_CPUS_PER_TASK:-8}"
done
