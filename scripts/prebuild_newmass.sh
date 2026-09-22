#!/bin/bash
# Prebuild ONLY the new internal-mass datasets (Z-mass + exotic top/Higgs/Z-4ℓ) with
# fiducial cuts on, into a dedicated cut-tagged cache. CPU prepost (weight 0, no GPU).
#SBATCH --job-name=prebuild_newmass
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=prebuild_newmass_%j.out
#SBATCH --error=prebuild_newmass_%j.err
set -euo pipefail
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"

# Dedicated cut-tagged cache (kept separate from the old pre-cut datasets_scanbig).
export AMP_TRAIN_CACHE_DIR=$SCRATCH/amp_cache_scanbig_cut
export AMP_FROZEN_DIR=$SCRATCH/datasets_scanbig_cut
export AMP_FIDUCIAL_CUTS=on
mkdir -p "$AMP_TRAIN_CACHE_DIR" "$AMP_FROZEN_DIR"

WORKERS="${SLURM_CPUS_PER_TASK:-48}"
python prebuild_recipes.py recipes/scan_bigrun_newmass.yaml --workers "$WORKERS" --auto-workers --seed 42
