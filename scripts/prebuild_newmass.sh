#!/bin/bash
# Prebuild ONLY the new internal-mass datasets (Z-mass + exotic top/Higgs/Z-4ℓ) with
# fiducial cuts on, into a dedicated cut-tagged cache. CPU prepost (weight 0, no GPU).
#SBATCH --job-name=prebuild_newmass
#SBATCH --partition=prepost
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=prebuild_newmass_%j.out
#SBATCH --error=prebuild_newmass_%j.err
set -euo pipefail
module load anaconda-py3/2023.09
source /gpfslocalsup/pub/anaconda-py3/2023.09/etc/profile.d/conda.sh
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

# Dedicated cut-tagged cache (kept separate from the old pre-cut datasets_scanbig).
export AMP_TRAIN_CACHE_DIR=$SCRATCH/amp_cache_scanbig_cut
export AMP_FROZEN_DIR=$SCRATCH/datasets_scanbig_cut
export AMP_FIDUCIAL_CUTS=on
mkdir -p "$AMP_TRAIN_CACHE_DIR" "$AMP_FROZEN_DIR"

WORKERS="${SLURM_CPUS_PER_TASK:-48}"
python prebuild_recipes.py recipes/scan_bigrun_newmass.yaml --workers "$WORKERS" --auto-workers --seed 42
