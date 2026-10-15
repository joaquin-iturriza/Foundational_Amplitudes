#!/bin/bash
#SBATCH --job-name=decomp_targets
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# CPU only: the decomposed targets (D48) of every role's pool of a recipe, as training resolves them
# (tools/decomp_targets.py --recipe); sidecars in ${DATA_DIR}/decomp, each checked against the stored |M|^2.
#     site submit <site> FA scripts/job_decomp_targets.sh -- <process> <recipe>
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1
cd "$PROJECT_DIR"
python tools/decomp_targets.py --process "$1" --recipe "$2" --workdir "$SCRATCH/decomp_sa"
