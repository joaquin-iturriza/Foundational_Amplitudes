#!/bin/bash
#SBATCH --job-name=catalog_v4_equality
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=catalog_v4_equality_%j.out
#SBATCH --error=catalog_v4_equality_%j.out
#
# CPU only (D130): catalog_v4's equality step, as catalog_v3's (scripts/job_catalog_v3_equality.sh): which census trees on
# their v4 strings (recipes/catalog_v4_census.yaml: full coupling orders, massless b) are the same dataset up to a constant
# and a relabelling (tools/equal_processes.py). The v4 classes can split (EW diagrams tell relabelled members apart) or
# merge (a massless b equals a d or s). Compiles the v4 standalone backends it lacks (shared MadGraph work dir, v4_*
# names, reused by the regeneration). Writes $SCRATCH/zz_catalog_v4_census/equality/equal_processes.json.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
N="${SLURM_CPUS_PER_TASK:-16}"
python tools/equal_processes.py --recipes recipes/catalog_v4_census.yaml --workers "$N" --compile-workers "$N" \
    --out "$SCRATCH/zz_catalog_v4_census/equality"
