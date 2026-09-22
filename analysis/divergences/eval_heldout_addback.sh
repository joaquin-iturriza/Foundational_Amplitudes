#!/bin/bash
#SBATCH --job-name=ho_eval_addback
#SBATCH --account=lpnhe
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=analysis/divergences/ho_eval_addback_%j.out
#SBATCH --error=analysis/divergences/ho_eval_addback_%j.out
#
# Regenerate the BASE ee->uug add-back eval (heldout_eval_ft_f<tag>.npz) for every add-back
# fraction. This is the companion of eval_heldout_studies.sh (uugg / soft / collinear), which
# was recorded while the base uug arm never was -- so when heldout_eval_ft_f100.npz was later
# overwritten by an unrelated eval (keys shrank to pred/true/sigma/y_min, losing `cut` and
# `x_gmin`) there was no recorded command to rebuild it, and both addback_curve and
# heldout_resid_f100 became unreproducible.
#
# All five tags are re-run together, not just f100, so every npz and the summary json share one
# provenance instead of mixing a July f000 with a fresh f100.
set -e
module purge
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes

python analysis/divergences/eval_heldout.py \
  --tags 000,005,015,050,100 \
  --heldout analysis/divergences/uug_heldtest.npz \
  --run_prefix ft_f --out_prefix heldout_eval_ft_f \
  --summary heldout_eval_summary.json --ckpt model_run0_best.pt

echo "DONE ho_eval_addback"
