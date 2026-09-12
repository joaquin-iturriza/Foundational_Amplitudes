#!/bin/bash
#SBATCH --job-name=recipe_test
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --account=lpnhe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
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
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes

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
