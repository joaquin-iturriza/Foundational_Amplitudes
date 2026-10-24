#!/bin/bash
#SBATCH --job-name=shape_learn
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=shape_learn_%j.out
#SBATCH --error=shape_learn_%j.out
#
# CPU only (D83): a shape proxy per candidate target, one small regressor per process on the 43 new processes (the 2->4 +g
# probes' val pools are too small and are listed as skipped) (analysis/catalog_daniel/shape_learn.py). Writes $SCRATCH/zz_shape_learn/shape_learn.json.

_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
python analysis/catalog_daniel/shape_learn.py --workers "${SLURM_CPUS_PER_TASK:-16}" "$@"
