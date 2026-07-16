#!/bin/bash
#SBATCH --job-name=twostage2
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=analysis/divergences/twostage2_%A_%a.out
#SBATCH --error=analysis/divergences/twostage2_%A_%a.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
WT=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-heterosc
MAIN=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
cd "$WT"

# STAGE 2 (sigma head on a frozen MSE mu) applied to the REAL targets, on a SHORT schedule.
#
# The 4000-step proof-of-concept (twostage_sigma.sh) converged in <800 iters and gave
# pull std 1.014 / reliability slope 1.08 / mu-MSE unchanged at 4.71e-5. Everything after ~800
# steps was just cosine-decaying an already-converged 258-param readout row. So: 600 iters at a
# 2x lr (4e-2 vs the 2.02e-2 the mu model was trained at). NOTE this is a DIFFERENT schedule,
# not a truncation of the old one (cosine is defined over the full horizon) -- arm 0 exists
# precisely to check the short schedule reproduces the 4000-step numbers before we trust arms
# 1-2. Single lr, chosen by hand, one value: NOT an lr grid.
#
#   arm 0  short_uug   SCHEDULE CHECK. Same mu model as the 4000-step run (the plain-MSE DyHPO
#                      best, mse_scratch_hpo/trial_0086, mu-MSE 4.72e-5, antenna uug from
#                      scratch). Must reproduce pull std ~1.0, slope ~1.1, mu-MSE 4.71e-5.
#   arm 1  base22      THE FOUNDATION. 22-process leave-uug-out MSE base (runs/
#                      pretrain22_heldout_uug/base, val combined mu-MSE 2.035e-2). Its 22
#                      per-process MSEs span 3.6e-7 .. 0.162 -- 4.5 DECADES -- so this is the
#                      real test of whether sigma tracks the error ACROSS processes, not just
#                      within one. sigma seeded at the geo-mean residual (0.0134).
#   arm 2  ft_antenna  THE DEEP-IR TARGET. The antenna-sampled uug finetune (ft_deep_antenna,
#                      mu-MSE 8.47e-5) -- the model where uncertainties in the singular region
#                      were the point of the whole divergences study. sigma seeded at 9.2e-3.
#
# mu is FROZEN in every arm (heterosc_sigma_only), so each arm's val mu-MSE MUST come back at
# its stage-1 value; that is the freeze's self-check and is reported in each log. Because mu
# cannot move, a sigma head can never damage the model it is attached to -- these are additive.
# beta=0: with mu fixed the plain Gaussian NLL is a proper scoring rule for sigma.

REC=$MAIN/recipes/pretrain22_heldout_uug.yaml

NAMES=(short_uug base22 ft_antenna)
CKPTS=(
  $WT/runs/mse_scratch_hpo/trial_0086/models/model_run0_best_2ch.pt
  $MAIN/runs/pretrain22_heldout_uug/base/models/model_run0_best_2ch.pt
  $MAIN/runs/pretrain22_heldout_uug/ft_deep_antenna/models/model_run0_best_2ch.pt
)

i=$SLURM_ARRAY_TASK_ID
NAME=${NAMES[$i]}; CKPT=${CKPTS[$i]}

# The array varies ONLY the warm-start checkpoint + its data source (a non-HP ablation axis).
# Every HP is a single constant across arms -- in particular regularization_lambda, which is
# FIXED at the value the validated 4000-step two-stage run used. It is deliberately NOT each
# stage-1 model's own tuned lambda: with the trunk frozen the L2 term reaches only the 258-param
# sigma readout row, so it is no longer "that model's lambda", just a tiny L2 on the sigma head,
# and it must be identical across arms for them to be comparable (inlined literal below).

# arm 1 is the 22-process recipe foundation; arms 0/2 are the single-dataset antenna uug pool.
if [ "$i" -eq 1 ]; then
  DATA_ARGS=(data.source=recipes data.processes_file="$REC")
else
  DATA_ARGS=(data.source=files data.data_path="$MAIN/data_deep_antenna/"
             'data.dataset=[ee_uug_91-1000GeV_amplitudes]'
             data.preprocess_per_dataset=true
             'data.train_test_val=[0.9, 0.05, 0.05]'
             data.subsample=null)
fi

python run.py \
  exp_name=heterosc_twostage2 run_name=${NAME} \
  "${DATA_ARGS[@]}" \
  model.use_diagrams=false model.particle_encoder_hidden=0 \
  model.net.num_blocks=8 model.net.num_heads=8 \
  model.net.detach_sigma_backbone=false \
  model.net.sigma_after_pool=true \
  fine_tune.pretrained_path="$CKPT" \
  fine_tune.reset_output_head=false \
  fine_tune.lr_scale=1.0 fine_tune.layer_decay=1.0 \
  training.loss=HETEROSC \
  training.heterosc_beta=0.0 \
  training.heterosc_sigma_only=true \
  training.lr=0.04 \
  training.iterations=600 training.batchsize=16384 \
  evaluation.batchsize=16384 \
  training.validate_frac=0.02 \
  training.clip_grad_norm=5 \
  training.regularization=L2 training.regularization_lambda=9.892346e-07 \
  training.scheduler=CosineAnnealingLR \
  training.cosanneal_warmup_frac=0.15 training.cosanneal_eta_min=1.0e-8 \
  plot=true save=true

echo "DONE ${NAME}"
