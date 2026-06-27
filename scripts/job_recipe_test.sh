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
# End-to-end smoke test of the on-the-fly recipe data path (source: recipes):
# cold generate -> train-only frozen stats -> train -> eval -> plot.
# Uses the two already-compiled backends (ee_uu, ee_uug) and small event counts.
# Isolated dirs so it neither pollutes nor reuses real data; clean up after.

set -euo pipefail
module load anaconda-py3/2023.09
# Source conda.sh so `conda activate` works in a non-interactive sbatch shell
# (a bare `conda activate` fails here: the conda shell function isn't loaded).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

# Isolated, disposable storage for this test
export AMP_FROZEN_DIR=$WORK/datasets_recipe_test
export AMP_TRAIN_CACHE_DIR=$SCRATCH/amp_data_cache_recipe_test

python run.py \
  exp_name=amp_recipe_test \
  data.source=recipes \
  data.seed=42 \
  '+data.processes=[{name:ee_uu,sqrts:[91,1000],n_train:5000,n_val:1000,n_test:1000},{name:ee_uug,sqrts:[91,1000],n_train:5000,n_val:1000,n_test:1000}]' \
  training.iterations=500 \
  training.batchsize=256 \
  training.validate_every_n_steps=100 \
  plot=true
