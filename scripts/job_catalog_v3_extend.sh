#!/bin/bash
#SBATCH --job-name=catalog_v3_extend
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=catalog_v3_extend_%j.out
#SBATCH --error=catalog_v3_extend_%j.out
#
# CPU only (D100): the 2->4 extension of the catalog_v3 census: the new candidates through MadGraph
# (gen_catalog_v3.py --check --cands recipes/catalog_v3_candidates_ext.json), the tree entries rewritten from both stage-2
# outputs, and the numeric equality over every tree (tools/equal_processes.py). Writes
# $SCRATCH/zz_catalog_v3_census/{checked_ext.json,equality_ext/}; the entries are written in the job's checkout only
# (brought home and committed from the laptop).

_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
N="${SLURM_CPUS_PER_TASK:-16}"; D="$SCRATCH/zz_catalog_v3_census"
python recipes/gen_catalog_v3.py --check --cands recipes/catalog_v3_candidates_ext.json --workers "$N" --out "$D/checked_ext.json"
python recipes/gen_catalog_v3.py --write "$D/checked.json" "$D/checked_ext.json"
python tools/equal_processes.py --recipes recipes/catalog_v3_census.yaml --workers "$N" --compile-workers "$N" --out "$D/equality_ext"
