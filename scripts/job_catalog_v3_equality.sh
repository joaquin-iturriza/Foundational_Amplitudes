#!/bin/bash
#SBATCH --job-name=catalog_v3_equality
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=catalog_v3_equality_%j.out
#SBATCH --error=catalog_v3_equality_%j.out
#
# CPU only (D98): the new catalog's census, stage 3: which of the census trees are the same dataset up to a constant and
# a relabelling (tools/equal_processes.py over recipes/catalog_v3_census.yaml; compiles the 1630 standalone backends,
# ~3 MB each), and the BSM candidates through MadGraph. Writes $SCRATCH/zz_catalog_v3_census/{equality,bsm_checked.json}.

_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
N="${SLURM_CPUS_PER_TASK:-16}"
python recipes/gen_catalog_v3.py --check --only-bsm --workers "$N" --out "$SCRATCH/zz_catalog_v3_census/bsm_checked.json"
python tools/equal_processes.py --recipes recipes/catalog_v3_census.yaml --workers "$N" --compile-workers "$N" \
    --out "$SCRATCH/zz_catalog_v3_census/equality"
