#!/bin/bash
#SBATCH --job-name=evalTrackA
#SBATCH --account=lpnhe
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=analysis/divergences/evalTrackA_%j.out
#SBATCH --error=analysis/divergences/evalTrackA_%j.out
set -e
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/worktrees/wt-harder-div
export PYTHONDONTWRITEBYTECODE=1
# Re-score the 4 completed Track A runs with the CORRECT process (the fold-in eval had used the
# default uugg NP on uug/uuggg momenta). Models are unchanged; this only re-runs the held-out eval.
E=analysis/divergences/eval_heldout_uugg.py
python $E --process uug   --run_dir runs/eeuu_l2uug/uug_base_s0     --label uug_base_s0
python $E --process uug   --run_dir runs/eeuu_l2uug/uug_sigma_s0    --label uug_g3_s0
python $E --process uuggg --run_dir runs/eeuu_l2uuggg/uuggg_base_s0  --label uuggg_base_s0
python $E --process uuggg --run_dir runs/eeuu_l2uuggg/uuggg_sigma_s0 --label uuggg_sigma_s0
echo "DONE eval Track A"
