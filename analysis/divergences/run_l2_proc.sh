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
# Run from the checkout this script was SUBMITTED from, not a hardcoded path: this file also lives
# in worktrees, and a hardcoded trunk path silently runs the TRUNK driver instead of the worktree's
# (which fails on any flag the trunk does not have yet, or worse, quietly runs the wrong code).
cd "${L2_ROOT:-${SLURM_SUBMIT_DIR:-/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes}}"
echo "[run_l2_proc] cwd=$(pwd)"
export PYTHONDONTWRITEBYTECODE=1
# Process-general L2 online-generation run (uug multi-scale Z-res+IR, uugg, uuggg more-legs).
# base arm = uniform keep; sigma arm = keep prop sigma^gamma. Held-out deep-IR eval folded into tail.
PROCESS=${PROCESS:-uug}; ARM=${ARM:-sigma}; SEED=${SEED:-0}; GAMMA=${GAMMA:-1.0}
STEPS=${STEPS:-4000}; NTOTAL=${NTOTAL:-300000}; ROUNDS=${ROUNDS:-10}
# SAT_LOG: per-round saturation diagnostics (sigma percentiles vs pool size) -> json.
# TAG lets an N-sweep keep its run dirs disjoint (run_name embeds the tag).
SAT_LOG=${SAT_LOG:-}; TAG=${TAG:-$PROCESS}
EXTRA=""; [ -n "$SAT_LOG" ] && EXTRA="--sat_log $SAT_LOG"
python analysis/divergences/l2_online_uugg.py \
  --process $PROCESS --arm $ARM --tag $TAG --seed $SEED $EXTRA \
  --total_steps $STEPS --n_total $NTOTAL --rounds $ROUNDS \
  --oversample ${OVERSAMPLE:-4} --gamma $GAMMA --y_lo 1e-6 --mix_ir 0.5 --sigma0 0.1 \
  --bbb_beta ${BBB_BETA:-1e-2} --bbb_sigma_rel ${BBB_SIGMA_REL:-0.05} --bbb_ksamples ${BBB_KSAMPLES:-16} \
  --heldout_eval
echo "DONE l2 process=$PROCESS arm=$ARM seed=$SEED gamma=$GAMMA n_total=$NTOTAL tag=$TAG"
