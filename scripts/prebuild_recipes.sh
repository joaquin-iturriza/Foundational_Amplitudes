#!/bin/bash
# Parallel CPU prebuild of recipe datasets — decoupled from GPU training.
#
# Runs on the `prepost` partition: CPU is billed at weight 0 (does NOT consume
# the V100/A100 GPU-hour allocation), up to 20h, 48 physical cores/node — so cores
# are effectively free. Cost-aware chunking (datagen.py) sizes each process's work
# units for ≈ equal wall-time, so an expensive 2→4 no longer becomes one fat chunk
# that the whole prebuild waits on; cores stay saturated and training starts sooner.
#
# Usage:
#   sbatch prebuild_recipes.sh recipes/pretrain8_D1e5.yaml [--seed 42] [--workers 48]
#
#SBATCH --job-name=prebuild_recipes
#SBATCH --partition=prepost
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#PORT #SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=prebuild_%j.out
#SBATCH --error=prebuild_%j.err

set -euo pipefail

SPEC="${1:?usage: sbatch prebuild_recipes.sh <spec.yaml> [extra args]}"
shift || true

module load anaconda-py3/2023.09 && source /gpfslocalsup/pub/anaconda-py3/2023.09/etc/profile.d/conda.sh
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
# One thread per process: every worker and every MadLoop/MG5 subprocess otherwise opens an
# OpenBLAS/OpenMP pool sized to the node (512 logical CPUs), and threads count against the
# per-user process limit (ulimit -u 1024) -> fork fails with EAGAIN mid-run.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

# Use the cores SLURM gave us for the worker pool.
WORKERS="${SLURM_CPUS_PER_TASK:-16}"

python prebuild_recipes.py "$SPEC" --workers "$WORKERS" --auto-workers "$@"
