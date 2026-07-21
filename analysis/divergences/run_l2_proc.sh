#!/bin/bash
#SBATCH --job-name=l2proc
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=analysis/divergences/l2proc_%x_%j.out
#SBATCH --error=analysis/divergences/l2proc_%x_%j.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-harder-div
export PYTHONDONTWRITEBYTECODE=1
# Process-general L2 online-generation run (uug multi-scale Z-res+IR, uugg, uuggg more-legs).
# base arm = uniform keep; sigma arm = keep prop sigma^gamma. Held-out deep-IR eval folded into tail.
PROCESS=${PROCESS:-uug}; ARM=${ARM:-sigma}; SEED=${SEED:-0}; GAMMA=${GAMMA:-1.0}
STEPS=${STEPS:-4000}; NTOTAL=${NTOTAL:-300000}; ROUNDS=${ROUNDS:-10}
python analysis/divergences/l2_online_uugg.py \
  --process $PROCESS --arm $ARM --tag ${TAG:-$PROCESS} --seed $SEED \
  --total_steps $STEPS --n_total $NTOTAL --rounds $ROUNDS \
  --oversample ${OVERSAMPLE:-4} --gamma $GAMMA --y_lo 1e-6 --mix_ir 0.5 --sigma0 0.1 \
  --bbb_beta ${BBB_BETA:-1e-2} --bbb_sigma_rel ${BBB_SIGMA_REL:-0.05} --bbb_ksamples ${BBB_KSAMPLES:-16} \
  --heldout_eval
echo "DONE l2 process=$PROCESS arm=$ARM seed=$SEED gamma=$GAMMA"
