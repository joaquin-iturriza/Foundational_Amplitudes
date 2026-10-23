#!/bin/bash
#SBATCH --job-name=norm_targets
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=norm_targets_%j.out
#SBATCH --error=norm_targets_%j.out
#
# CPU only (D83): the per-event scales of every candidate target on the catalog's pools, the 2->4 probes and the
# one-loop probes (analysis/catalog_daniel/norm_targets_stats.py). Writes $SCRATCH/zz_norm_targets/norm_targets_stats.json.

_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
python analysis/catalog_daniel/norm_targets_stats.py --workers "${SLURM_CPUS_PER_TASK:-16}" "$@"
