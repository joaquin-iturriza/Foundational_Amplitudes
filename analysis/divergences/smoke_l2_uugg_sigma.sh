#!/bin/bash
#SBATCH --job-name=smoke_l2sig
#SBATCH --account=lpnhe
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=analysis/divergences/smoke_l2sig_%j.out
#SBATCH --error=analysis/divergences/smoke_l2sig_%j.out
set -e
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/worktrees/wt-l2-uugg
export PYTHONDONTWRITEBYTECODE=1
python analysis/divergences/l2_online_uugg.py \
  --arm sigma --tag smoke --total_steps 12 --rounds 3 --n_total 1500 \
  --oversample 6 --gamma 1.0 --y_lo 1e-6 --mix_ir 0.5 --seed 0
echo "DONE smoke sigma"
