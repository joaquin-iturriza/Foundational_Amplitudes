#!/bin/bash
#SBATCH --job-name=twostage
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=analysis/divergences/twostage_%A_%a.out
#SBATCH --error=analysis/divergences/twostage_%A_%a.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
WT=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-heterosc
MAIN=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
cd "$WT"

# TWO-STAGE HETEROSC. Every coupled (mu,sigma) run so far has failed for one of two reasons,
# both now understood analytically:
#   (a) beta>0: profiling sigma out at its stationary point (sigma=|r|) leaves
#       L(r)=r^{2beta}(1/2+ln r), minimised at r*=exp(-1/2-1/(2beta)) -- NOT at r=0. At beta=1
#       that is r*=1/e: a 37%-error model scores BETTER than a perfect one. The beta=1 uug
#       finetune duly parked at mu-MSE 0.086 (r=0.29), sigma=0.30, pull_std 0.93.
#   (b) beta=0: the 1/sigma^2 factor starves mu's gradient (Seitzer's sigma-inflation).
# Both are artifacts of fitting mu and sigma JOINTLY. Fix: don't.
#
# Stage 1 (DONE, reused per the A/B protocol -- no baseline is retrained): the best trial of the
# plain-MSE DyHPO sweep, mse_scratch_hpo/trial_0086, val mu-MSE 4.72e-5.
# Stage 2 (here): grow_sigma_head.py added a sigma row to that readout (mu row verbatim, sigma
# seeded at softplus^-1(6.9e-3) = the model's own RMS residual). Now FREEZE trunk+mu and train
# ONLY the sigma row with the plain Gaussian NLL (beta=0), which with mu fixed is a PROPER
# scoring rule for sigma. No multi-task coupling, no spurious optimum.
#
#   arm 0  sigma_only=true   two-stage. mu is frozen => val mu-MSE MUST come back 4.72e-5
#                            (that is also the freeze's self-check). Question: is sigma
#                            calibrated (pull_std ~1, reliability slope ~1)?
#   arm 1  sigma_only=false  CONTROL: same beta=0 NLL, same MSE-grown init, but everything
#                            trainable. Isolates the freeze as the ONLY variable. Expect mu to
#                            DEGRADE away from 4.72e-5 if joint fitting is really the problem.
#
# Both arms use the stage-1 trial's own HPs (A/B protocol step 3: try the baseline's best HPs
# first). Checkpoint selection is the fixed one (base_experiment): mu-MSE when mu is trainable
# (arm 1), the beta=0 NLL when mu is frozen (arm 0, where mu-MSE is constant by construction).

STAGE1=$WT/runs/mse_scratch_hpo/trial_0086
CKPT=$STAGE1/models/model_run0_best_2ch.pt        # written by grow_sigma_head.py
DATA_PATH=$MAIN/data_deep_antenna/

NAMES=(sigma_only joint_ctrl)
SIGONLY=(true      false)
i=$SLURM_ARRAY_TASK_ID
NAME=${NAMES[$i]}; SO=${SIGONLY[$i]}

python run.py \
  exp_name=heterosc_twostage run_name=${NAME} \
  data.source=files data.data_path="$DATA_PATH" \
  'data.dataset=[ee_uug_91-1000GeV_amplitudes]' \
  data.preprocess_per_dataset=true \
  'data.train_test_val=[0.9, 0.05, 0.05]' \
  data.subsample=null \
  model.use_diagrams=false model.particle_encoder_hidden=0 \
  model.net.num_blocks=8 model.net.num_heads=8 \
  model.net.detach_sigma_backbone=false \
  model.net.sigma_after_pool=true \
  fine_tune.pretrained_path="$CKPT" \
  fine_tune.reset_output_head=false \
  fine_tune.lr_scale=1.0 fine_tune.layer_decay=1.0 \
  training.loss=HETEROSC \
  training.heterosc_beta=0.0 \
  training.heterosc_sigma_only=${SO} \
  training.lr=0.0202319 \
  training.iterations=4000 training.batchsize=16384 \
  evaluation.batchsize=16384 \
  training.clip_grad_norm=5 \
  training.regularization=L2 training.regularization_lambda=9.892346e-07 \
  training.scheduler=CosineAnnealingLR \
  training.cosanneal_warmup_frac=0.1537129 training.cosanneal_eta_min=1.0e-8 \
  plot=true save=true

echo "DONE ${NAME}"
