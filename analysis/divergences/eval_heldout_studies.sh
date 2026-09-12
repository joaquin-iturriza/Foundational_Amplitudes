#!/bin/bash
#SBATCH --job-name=ho_eval_studies
#SBATCH --account=lpnhe
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=analysis/divergences/ho_eval_studies_%j.out
#SBATCH --error=analysis/divergences/ho_eval_studies_%j.out

set -e
module purge
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes

E=analysis/divergences/eval_heldout.py

echo "===== uugg add-back eval ====="
python $E --tags 000,005,015,050,100 \
  --heldout analysis/divergences/uugg_heldtest.npz \
  --run_prefix ft_uugg_f --out_prefix heldout_eval_uugg_f \
  --summary heldout_eval_uugg_summary.json --ckpt model_run0_best.pt

echo "===== uug SOFT-cut eval ====="
python $E --tags 000,005,015,100 \
  --heldout analysis/divergences/uug_soft_heldtest.npz \
  --run_prefix ft_soft_f --out_prefix heldout_eval_soft_f \
  --summary heldout_eval_soft_summary.json --ckpt model_run0_best.pt

echo "===== uug COLLINEAR-cut eval ====="
python $E --tags 000,005,015,100 \
  --heldout analysis/divergences/uug_coll_heldtest.npz \
  --run_prefix ft_coll_f --out_prefix heldout_eval_coll_f \
  --summary heldout_eval_coll_summary.json --ckpt model_run0_best.pt

echo "DONE ho_eval_studies"
