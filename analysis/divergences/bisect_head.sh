#!/bin/bash
#SBATCH --job-name=bisect
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=analysis/divergences/bisect_%A_%a.out
#SBATCH --error=analysis/divergences/bisect_%A_%a.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
WT=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-heterosc
MAIN=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
cd "$WT"

# BISECTION: is the mu underfit caused by the NLL OBJECTIVE or by the 2-ch MODEL/wiring?
# The loss math is verified correct (autograd: beta=1 gives grad_mu exactly 0.5x the MSE
# gradient, uniformly per event -> Adam-invariant). So a 2-ch net trained with plain MSE on
# mu MUST reproduce the 1-ch MSE result (mse_ref = 1.11e-4) unless the MODEL is at fault.
#   arm 0 mu_only : 2-ch HETEROSC net, loss = plain MSE on mu (sigma gets NO gradient)
#                   -> 1.1e-4 : model is fine, the sigma term/gradient is the cause
#                   -> ~0.03-0.09 : the 2-ch MODEL or its warm-start/param-groups is the BUG
#   arm 1 full_b1 : full HETEROSC beta=1 (reference, expect ~0.086)
# Everything else matches mse_ref exactly: same HETEROSC foundation body, lr_scale 0.339,
# layer_decay 0.999, clip 5, 4000 steps, antenna uug.
NAMES=(het_mu_only het_full_b1)
MUONLY=(true       false)
i=$SLURM_ARRAY_TASK_ID
NAME=${NAMES[$i]}; MO=${MUONLY[$i]}

CKPT=$WT/runs/heterosc_foundation/het_lr0.004_b1.0/models/model_run0_best.pt
DATA_PATH=$MAIN/data_deep_antenna/

python run.py \
  exp_name=heterosc_bisect run_name=${NAME} \
  data.source=files data.data_path="$DATA_PATH" \
  'data.dataset=[ee_uug_91-1000GeV_amplitudes]' \
  data.preprocess_per_dataset=true \
  'data.train_test_val=[0.9, 0.05, 0.05]' \
  data.subsample=null \
  model.use_diagrams=false model.particle_encoder_hidden=0 \
  model.net.num_blocks=8 model.net.num_heads=8 \
  model.net.detach_sigma_backbone=false \
  fine_tune.pretrained_path="$CKPT" \
  fine_tune.lr_scale=0.339 fine_tune.layer_decay=0.999 \
  training.loss=HETEROSC training.heterosc_beta=1.0 \
  training.heterosc_mu_only=${MO} \
  training.clip_grad_norm=5 \
  training.lr=0.004 training.iterations=4000 training.batchsize=16384 \
  training.regularization=L2 training.regularization_lambda=2.47e-7 \
  training.scheduler=CosineAnnealingLR \
  training.cosanneal_warmup_frac=0.191 training.cosanneal_eta_min=1.6e-7 \
  plotting.plot_mse_het=false plot=true save=true

echo "DONE ${NAME}"
