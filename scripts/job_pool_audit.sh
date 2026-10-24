#!/bin/bash
#SBATCH --job-name=pool_audit
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=pool_audit_%j.out
#SBATCH --error=pool_audit_%j.out
#
# CPU only: audit of catalog_v3's generated pools (analysis/catalog_v3/pool_audit.py, stages 1, 2a, 3)
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
python analysis/catalog_v3/pool_audit.py --workers 8 --out "$SCRATCH/zz_pool_audit.json"
