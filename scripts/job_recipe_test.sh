#!/bin/bash
#SBATCH --job-name=recipe_test
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# End-to-end smoke test of the recipe data path (source: recipes) against a
# PREBUILT recipe: train-only frozen stats -> train -> eval -> plot. Training jobs
# never generate (data.require_cache: true), so run the CPU prebuild first:
#   scripts/remote.sh sbatch scripts/prebuild_recipes.sh recipes/pretrain8_short.yaml
# Reads the real pools ($WORK/datasets, $SCRATCH/amp_data_cache via env_ccin2p3.sh).

set -euo pipefail
module load anaconda-py3/2023.09 && source /gpfslocalsup/pub/anaconda-py3/2023.09/etc/profile.d/conda.sh
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

RECIPE="${RECIPE:-recipes/pretrain8_short.yaml}"

python run.py \
  exp_name=amp_recipe_test \
  data.source=recipes \
  data.seed=42 \
  data.processes_file="$RECIPE" \
  training.iterations=500 \
  training.batchsize=256 \
  training.validate_every_n_steps=100 \
  plot=true
