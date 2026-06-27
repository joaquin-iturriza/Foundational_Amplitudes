#!/bin/bash
# Parallel CPU prebuild of recipe datasets — decoupled from GPU training.
#
# Runs on the `prepost` partition: CPU is billed at weight 0 (does NOT consume
# the V100/A100 GPU-hour allocation), up to 20h, many cores per node.
#
# Usage:
#   sbatch prebuild_recipes.sh recipes/pretrain8_D1e5.yaml [--seed 42] [--workers 32]
#
#SBATCH --job-name=prebuild_recipes
#SBATCH --partition=prepost
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=prebuild_%j.out
#SBATCH --error=prebuild_%j.err

set -euo pipefail

SPEC="${1:?usage: sbatch prebuild_recipes.sh <spec.yaml> [extra args]}"
shift || true

module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational

cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

# Use the cores SLURM gave us for the worker pool.
WORKERS="${SLURM_CPUS_PER_TASK:-16}"

python prebuild_recipes.py "$SPEC" --workers "$WORKERS" "$@"
