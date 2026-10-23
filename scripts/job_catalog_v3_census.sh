#!/bin/bash
#SBATCH --job-name=catalog_v3_census
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=catalog_v3_census_%j.out
#SBATCH --error=catalog_v3_census_%j.out
#
# CPU only (D98): the new catalog's census, stage 2: every enumerated tree and BSM candidate through MadGraph
# (recipes/gen_catalog_v3.py --check). Writes $SCRATCH/zz_catalog_v3_census/checked.json. No events.

_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
python recipes/gen_catalog_v3.py --check --workers "${SLURM_CPUS_PER_TASK:-16}" "$@"
