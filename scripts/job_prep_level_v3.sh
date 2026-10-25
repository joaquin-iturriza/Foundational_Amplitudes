#!/bin/bash
#SBATCH --job-name=prep_level_v3
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=prep_level_v3_%j.out
#SBATCH --error=prep_level_v3_%j.out
#
# CPU only (the preprocessing exploration on catalog_v3, D83/D114): the class-mean level a target leaves for unseen classes (the hybrid design), from
# every training tree class and every held-out tree axis (analysis/catalog_v3/prep_level.py).
# Writes $SCRATCH/zz_v3_prep/prep_level.json.

_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
python analysis/catalog_v3/prep_level.py --workers "${SLURM_CPUS_PER_TASK:-32}" "$@"
