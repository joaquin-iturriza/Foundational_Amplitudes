#!/bin/bash
#SBATCH --job-name=ev_deep
#SBATCH --account=lpnhe
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=analysis/divergences/ev_deep_%j.out
#SBATCH --error=analysis/divergences/ev_deep_%j.out

set -e
module purge
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes

# Forward the COMMON deep-IR held-out test through both fine-tunes (uniform vs antenna),
# recovering each run's own preprocessing stats. Deep-IR-binned MSE(log|M|^2) is on the
# same absolute (de-standardized) scale, so the two samplings compare like-for-like.
python analysis/divergences/eval_heldout.py \
  --runs_root runs/pretrain22_heldout_uug \
  --run_prefix ft_deep_ --tags ${TAGS:-uniform,antenna,mixture} \
  --heldout analysis/divergences/uug_deep_test.npz \
  --out_prefix deep_eval_ \
  --summary deep_eval_summary.json

echo "DONE eval_deep"
