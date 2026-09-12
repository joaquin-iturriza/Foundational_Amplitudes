#!/bin/bash
#SBATCH --job-name=l2uugg
#SBATCH --account=lpnhe
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=analysis/divergences/l2uugg_%x_%j.out
#SBATCH --error=analysis/divergences/l2uugg_%x_%j.out
set -e
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/worktrees/wt-l2-uugg
export PYTHONDONTWRITEBYTECODE=1
ARM=${ARM:-sigma}; SEED=${SEED:-0}; TAG=${TAG:-run}
python analysis/divergences/l2_online_uugg.py \
  --arm $ARM --tag $TAG --total_steps 4000 --n_total 300000 --rounds 10 \
  --oversample ${OVERSAMPLE:-4} --gamma ${GAMMA:-1.0} --y_lo 1e-6 --mix_ir 0.5 --sigma0 0.1 --seed $SEED \
  --heldout_eval
echo "DONE l2uugg arm=$ARM seed=$SEED"
