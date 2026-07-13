#!/bin/bash
#SBATCH --job-name=clip_test
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=0-4
#SBATCH --output=analysis/divergences/clip_%A_%a.out
#SBATCH --error=analysis/divergences/clip_%A_%a.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
WT=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-heterosc
MAIN=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
cd "$WT"

# CLIP TEST. The isolation test showed the HETEROSC FOUNDATION is fine (MSE loss on its body
# -> 1.2e-4, ~= the 8.4e-5 MSE-foundation control). So the residual mu gap belongs to the
# heteroscedastic LOSS. But it is NOT (mu,sigma) backbone interference either: the detached-
# sigma run still underfit (0.031 vs MSE's 1.2e-4 at the SAME lr_scale=0.339 and same body).
# What remains is a coupling on the UPDATE, not the representation: clip_grad_norm is a GLOBAL
# norm over all params, so sigma's gradients inflate it and the clip scales the WHOLE update
# down -- mu included. (Consistent with the clean HPO wanting a big lr_scale=5.7 to compensate.)
#
# All arms: HETEROSC foundation body, antenna uug, 4000 steps, layer_decay=0.999.
# Arms 0-3 hold lr_scale=0.339 (the MSE reference's HP) so they compare directly to 1.2e-4.
#   0 het_clip5    : coupled, clip=5    -> reproduce the underfit (~0.086) + log its grad norm
#   1 het_clip50   : coupled, clip=50
#   2 het_clipoff  : coupled, clip=1e9  -> KEY: if clip is the throttle, expect ~1e-4
#   3 het_clipoff_lrs : coupled, clip=1e9 + lr_scale=5.716 (HPO best) -> best-case coupled
#   4 mse_ref      : loss=MSE, clip=5   -> the reference (1.2e-4) + ITS grad norm, to show
#                    whether clip binds for MSE at all (mechanism evidence)
NAMES=(het_clip5      het_clip50    het_clipoff   het_clipoff_lrs  mse_ref)
CLIPS=(5              50            1e9           1e9              5)
LRS=(0.339           0.339          0.339         5.716201         0.339)
LOSSES=(HETEROSC      HETEROSC      HETEROSC      HETEROSC         MSE)

i=$SLURM_ARRAY_TASK_ID
NAME=${NAMES[$i]}; CLIP=${CLIPS[$i]}; LRS_I=${LRS[$i]}; LOSS=${LOSSES[$i]}

# MSE arm needs the mu-sliced 1-ch checkpoint; HETEROSC arms use the native 2-ch one.
if [ "$LOSS" = "MSE" ]; then
  CKPT=$WT/runs/heterosc_foundation/het_lr0.004_b1.0/models/model_run0_best_mu1ch.pt
  EXTRA="training.loss_aggregation=geometric_mean"
else
  CKPT=$WT/runs/heterosc_foundation/het_lr0.004_b1.0/models/model_run0_best.pt
  EXTRA="training.heterosc_beta=1.0 model.net.detach_sigma_backbone=false"
fi
DATA_PATH=$MAIN/data_deep_antenna/

python run.py \
  exp_name=heterosc_clip run_name=${NAME} \
  data.source=files data.data_path="$DATA_PATH" \
  'data.dataset=[ee_uug_91-1000GeV_amplitudes]' \
  data.preprocess_per_dataset=true \
  'data.train_test_val=[0.9, 0.05, 0.05]' \
  data.subsample=null \
  model.use_diagrams=false model.particle_encoder_hidden=0 \
  model.net.num_blocks=8 model.net.num_heads=8 \
  fine_tune.pretrained_path="$CKPT" \
  fine_tune.lr_scale=${LRS_I} fine_tune.layer_decay=0.999 \
  training.loss=${LOSS} \
  training.clip_grad_norm=${CLIP} \
  training.lr=0.004 training.iterations=4000 training.batchsize=16384 \
  training.regularization=L2 training.regularization_lambda=2.47e-7 \
  training.scheduler=CosineAnnealingLR \
  training.cosanneal_warmup_frac=0.191 training.cosanneal_eta_min=1.6e-7 \
  ${EXTRA} \
  plot=true save=true

echo "DONE ${NAME}"
