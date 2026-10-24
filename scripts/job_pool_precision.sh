#!/bin/bash
#SBATCH --job-name=pool_precision
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=pool_precision_%j.out
#SBATCH --error=pool_precision_%j.out
#
# CPU only: can catalog_v3's pools be float32 (analysis/catalog_v3/pool_precision.py over the catalog val pools)
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
python analysis/catalog_v3/pool_precision.py --out "$SCRATCH/zz_pool_precision.json"
