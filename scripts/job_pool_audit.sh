#!/bin/bash
#SBATCH --job-name=pool_audit
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=pool_audit_%j.out
#SBATCH --error=pool_audit_%j.out
#
# CPU only: audit of catalog_v3's generated pools (analysis/catalog_v3/pool_audit.py, stages 1, 2a, 3); catalog_v4's with
# AUDIT_STAGES="v4_stage1 v4_stage2 v4_stage3 v4_stage4" AUDIT_OUT="\$SCRATCH/zz_pool_audit_v4.json" (site submit --env)
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
python analysis/catalog_v3/pool_audit.py --workers 8 --out "${AUDIT_OUT:-$SCRATCH/zz_pool_audit.json}" \
    ${AUDIT_STAGES:+--stages $AUDIT_STAGES}
