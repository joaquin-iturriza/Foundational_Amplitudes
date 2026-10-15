#!/bin/bash
#SBATCH --job-name=memprof
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=memprof_%j.out
#SBATCH --error=memprof_%j.out
#
# CPU only: host-memory profile of a sweep's data pipeline (tools/mem_profile_data.py): init_data, the
# dataloaders and a few hundred training batches through 2 workers, RSS sampled throughout. No GPU, no training.
#   site submit ccin2p3 FA scripts/job_mem_profile.sh --hdr=--mem=64G -- --sweep-config ... --n-procs 300 --out ...
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
python tools/mem_profile_data.py "$@"
