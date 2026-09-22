#!/bin/bash
#SBATCH --job-name=bbbRescore
#SBATCH --account=lpnhe
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=analysis/divergences/bbbRescore_%j.out
#SBATCH --error=analysis/divergences/bbbRescore_%j.out
set -e
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/worktrees/wt-harder-div
export PYTHONDONTWRITEBYTECODE=1
# Re-score the completed uugg bbb runs THROUGH the posterior (predictive mean over K samples +
# calibration), superseding the deterministic-mean fold-in numbers. --mc_samples auto-set by the
# variational-checkpoint detector.
E=analysis/divergences/eval_heldout_uugg.py
python $E --process uugg --run_dir runs/eeuu_l2uugg/uugg_bbb_s0    --label uugg_bbb_s0    --mc_samples 32
python $E --process uugg --run_dir runs/eeuu_l2uugg/uugg_bbb_g3_s0 --label uugg_bbb_g3_s0 --mc_samples 32
echo "DONE bbb re-score"
