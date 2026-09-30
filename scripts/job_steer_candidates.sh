#!/bin/bash
#SBATCH --job-name=steer_candidates
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Label the flat candidates of a sigma-steered pool on CPU (tools/steer_pool.py --stage candidates),
# cached under $SCRATCH/steer_candidates, before the GPU selection job.
#     site submit <site> FA scripts/job_steer_candidates.sh -- recipes/<steered recipe>.yaml
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
python tools/steer_pool.py --recipe "$1" --stage candidates
