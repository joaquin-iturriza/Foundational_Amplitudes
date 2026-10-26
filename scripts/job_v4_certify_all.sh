#!/bin/bash
#SBATCH --job-name=v4_certify_all
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=v4_certify_all_%j.out
#SBATCH --error=v4_certify_all_%j.out
#
# CPU only (catalog_coverage): the pole certification of every catalog_v4 one-loop entry in one list
# (tools/certify_v4_loops.py). Works in $SCRATCH/zz_v4_certify_all: built production backends are copied there,
# missing ones built there, so the running regeneration's backends are never touched.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
df -k "$SCRATCH" | tail -1 | awk '{print "CERT FREE_KB_START", $4}'
python tools/certify_v4_loops.py --workers "${SLURM_CPUS_PER_TASK:-16}" ${CERT_ARGS:-}
du -sk "$SCRATCH/zz_v4_certify_all" | awk '{print "CERT SIZE_KB", $1}'
