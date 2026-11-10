#!/bin/bash
#SBATCH --job-name=het_found
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:45:00
#SBATCH --array=0-5
#SBATCH --output=analysis/divergences/het_found_%A_%a.out
#SBATCH --error=analysis/divergences/het_found_%A_%a.out

set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
WT=$PROJECT_DIR
cd "$WT"   # run the HETEROSC-wired code

REC=$PROJECT_DIR/recipes/pretrain22_heldout_uug.yaml

# HETEROSC foundation mini-sweep: is the earlier mu-underfit an HP/loss-scale issue (your
# hypothesis) rather than fundamental? Replicate base22 (same recipe/data/arch/reg) but
# loss=HETEROSC, sweeping lr x beta-NLL. beta=1 makes mu's gradient ~ MSE's (lr~4e-3
# principled); beta=0 is plain NLL (sigma-inflation risk). 6000 iters is enough to rank
# mu-fit (the failed reset-head run underfit by 4000). Rank post-hoc by mu-MSE on uug.
LRS=(0.004 0.004 0.004 0.016 0.016 0.016)
BETAS=(0.0 0.5 1.0 0.0 0.5 1.0)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}
TAG=lr${LR}_b${BETA}

python run.py \
  exp_name=heterosc_foundation run_name=het_${TAG} \
  data.source=recipes data.processes_file="$REC" \
  model.use_diagrams=false model.particle_encoder_hidden=0 \
  training.loss=HETEROSC training.heterosc_beta=${BETA} \
  training.batchsize=16384 training.iterations=6000 training.lr=${LR} \
  training.regularization_lambda=6.264093e-10 \
  training.cosanneal_warmup_frac=0.1079041 training.cosanneal_eta_min=6.17785e-9 \
  training.ema_decay=0.9330307 \
  \
  plot=true save=true

echo "DONE het_${TAG}"
