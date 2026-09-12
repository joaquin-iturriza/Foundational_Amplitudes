#!/bin/bash
# Prebuild ONLY the new internal-mass datasets (Z-mass + exotic top/Higgs/Z-4ℓ) with
# fiducial cuts on, into a dedicated cut-tagged cache. CPU prepost (weight 0, no GPU).
#SBATCH --job-name=prebuild_newmass
#SBATCH --partition=htc
#SBATCH --account=lpnhe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=prebuild_newmass_%j.out
#SBATCH --error=prebuild_newmass_%j.err
set -euo pipefail
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes

# Dedicated cut-tagged cache (kept separate from the old pre-cut datasets_scanbig).
export AMP_TRAIN_CACHE_DIR=$SCRATCH/amp_cache_scanbig_cut
export AMP_FROZEN_DIR=$SCRATCH/datasets_scanbig_cut
export AMP_FIDUCIAL_CUTS=on
mkdir -p "$AMP_TRAIN_CACHE_DIR" "$AMP_FROZEN_DIR"

WORKERS="${SLURM_CPUS_PER_TASK:-48}"
python prebuild_recipes.py recipes/scan_bigrun_newmass.yaml --workers "$WORKERS" --auto-workers --seed 42
