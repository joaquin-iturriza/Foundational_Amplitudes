#!/bin/bash
#SBATCH --job-name=attrib_ig
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=attrib_ig_%j.out
#SBATCH --error=attrib_ig_%j.err
#SBATCH --gres=gpu:1
#
# Integrated-Gradients input importance for a trained LLoCa-μP model.
# Needs a GPU (xformers attention is CUDA-only). SUBMIT WITH: sbatch job_attribution.sh

_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"

python attribution_inputs.py \
  --run-dir runs/pretrain_full_nh4_fresh/trial_0266 \
  --ckpt model_run0_best.pt.gz \
  --frame aug \
  --n-per-process 256 \
  --ref-mode background \
  --budget 4096 \
  --mb-events 64 \
  --out-prefix runs/pretrain_full_nh4_fresh/trial_0266/attribution/final \
  --seed 0
