#!/bin/bash
#SBATCH --job-name=equal_procs
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=equal_procs_%j.out
#SBATCH --error=equal_procs_%j.out
#
# CPU only: which catalog trees are the same dataset up to a constant and a relabelling
# (tools/equal_processes.py). Writes analysis/process_equality/equal_processes.json.

_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
python tools/equal_processes.py --workers "${SLURM_CPUS_PER_TASK:-16}" "$@"
