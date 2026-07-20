#!/bin/bash
#SBATCH --job-name=l2uugg
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=analysis/divergences/l2uugg_%x_%j.out
#SBATCH --error=analysis/divergences/l2uugg_%x_%j.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-l2-uugg
export PYTHONDONTWRITEBYTECODE=1
ARM=${ARM:-sigma}; SEED=${SEED:-0}; TAG=${TAG:-run}
python analysis/divergences/l2_online_uugg.py \
  --arm $ARM --tag $TAG --total_steps 4000 --n_total 300000 --rounds 10 \
  --oversample ${OVERSAMPLE:-4} --gamma ${GAMMA:-1.0} --y_lo 1e-6 --mix_ir 0.5 --sigma0 0.1 --seed $SEED \
  --heldout_eval
echo "DONE l2uugg arm=$ARM seed=$SEED"
