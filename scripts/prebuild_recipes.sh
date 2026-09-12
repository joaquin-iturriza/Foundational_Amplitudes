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
#SBATCH --partition=htc
#SBATCH --account=lpnhe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=prebuild_%j.out
#SBATCH --error=prebuild_%j.err

set -euo pipefail

SPEC="${1:?usage: sbatch prebuild_recipes.sh <spec.yaml> [extra args]}"
shift || true

source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh

cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes

# Use the cores SLURM gave us for the worker pool.
WORKERS="${SLURM_CPUS_PER_TASK:-16}"

python prebuild_recipes.py "$SPEC" --workers "$WORKERS" --auto-workers "$@"
