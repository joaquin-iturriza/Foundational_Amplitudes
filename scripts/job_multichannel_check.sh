#!/bin/bash
#SBATCH --job-name=cov_v4_mcheck
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=cov_v4_mcheck_%j.out
#SBATCH --error=cov_v4_mcheck_%j.out
#
# CPU only: plan catalog_coverage step 57's check of the multichannel flat-part candidates on one class
# (analysis/catalog_v4/multichannel_check.py run: current and multichannel, seeds 42 and 43, the class's train pool
# replayed chunk by chunk into temporary dirs; nothing written to $DATA_DIR) -> $SCRATCH/zz_mcheck/<class>.json.
# MCHECK_CLASS and MCHECK_PROBE name the class and its PP4 probe (multichannel_check.CLASSES); MCHECK_POLICIES (e.g.
# multichannel+quota, plan step 58) and MCHECK_SEED run those policies at one seed -> <class>_<policies>_<seed>.json.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
: "${MCHECK_CLASS:?MCHECK_CLASS not set}" "${MCHECK_PROBE:?MCHECK_PROBE not set}"
if [ -n "${MCHECK_POLICIES:-}" ]; then
    : "${MCHECK_SEED:?MCHECK_SEED not set with MCHECK_POLICIES}"
    python analysis/catalog_v4/multichannel_check.py run "$MCHECK_CLASS" "$MCHECK_PROBE" "$MCHECK_POLICIES" "$MCHECK_SEED"
    TAG="${MCHECK_CLASS}_${MCHECK_POLICIES}_${MCHECK_SEED}"
else
    python analysis/catalog_v4/multichannel_check.py run "$MCHECK_CLASS" "$MCHECK_PROBE"
    TAG="$MCHECK_CLASS"
fi
echo "MCHECK_JSON $(cat "$SCRATCH/zz_mcheck/$TAG.json" | wc -c) bytes"
