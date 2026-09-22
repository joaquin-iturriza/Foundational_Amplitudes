#!/bin/bash
#SBATCH --job-name=recipe_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#
# End-to-end smoke test of the recipe data path (source: recipes) against a
# PREBUILT recipe: train-only frozen stats -> train -> eval -> plot. Training jobs
# never generate (data.require_cache: true), so run the CPU prebuild first:
#   site submit <site> FA scripts/prebuild_recipes.sh recipes/pretrain8_short.yaml
# Reads the real pools ($WORK/datasets, $SCRATCH/amp_data_cache via sites/activate.sh).

set -euo pipefail
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"

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
