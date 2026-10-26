#!/bin/bash
#SBATCH --job-name=prep_sampling_v3
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=prep_sampling_v3_%j.out
#SBATCH --error=prep_sampling_v3_%j.out
#
# CPU only (the preprocessing exploration on catalog_v3, D83/D114): the sampling sensitivity, each pool reweighted flat
# in log s_hat and towards its singular region, per target on the data and per short arm on its saved zero-shot
# predictions (analysis/catalog_v3/prep_sampling_pools.py). One SAMPLING line per class in the log; nothing written to disk.

_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
python analysis/catalog_v3/prep_sampling_pools.py --workers "${SLURM_CPUS_PER_TASK:-32}" "$@"
