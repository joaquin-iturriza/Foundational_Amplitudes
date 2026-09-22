#!/bin/bash
#SBATCH --job-name=attrib_ig
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --account=lpnhe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=attrib_ig_%j.out
#SBATCH --error=attrib_ig_%j.err
#
# Integrated-Gradients input importance for a trained LLoCa-μP model.
# Needs a GPU (xformers attention is CUDA-only). SUBMIT WITH: sbatch job_attribution.sh

source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes

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
