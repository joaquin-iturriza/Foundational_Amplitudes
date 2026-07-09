#!/bin/bash
#SBATCH --job-name=ft_uug_ho
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=0-4
#SBATCH --output=analysis/divergences/ft_uug_ho_%A_%a.out
#SBATCH --error=analysis/divergences/ft_uug_ho_%A_%a.out

set -e
module purge
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

REPO=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
CKPT=$REPO/runs/pretrain22_heldout_uug/base/models/model_run0_best.pt

# Add-back fractions f = {0, 0.05, 0.15, 0.5, 1.0} -> dir tags {000,005,015,050,100}.
TAGS=(000 005 015 050 100)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
DATA_PATH=$REPO/data_ho_f${TAG}/

# Fine-tune the leave-uug-out base22 foundation onto each add-back training set.
# HPs reuse the best trial of finetune_scaling_virt_002 (eeuu-virt, t=4000 cell):
#   lr_scale=0.339, layer_decay=0.999, reg_lambda=2.47e-7, warmup=0.191, eta_min=1.6e-7.
# lr anchor: the reference's finetune/pretrain LR RATIO was 0.339 (base lr 1.08e-3 *
#   lr_scale 0.339). We preserve that RATIO against base22's own pretrain lr (4e-3),
#   i.e. training.lr=4e-3 * lr_scale=0.339 -> effective ~1.36e-3 (rule #4: finetune lr
#   tracks pretrain lr). Arch overrides MUST match the base so the checkpoint loads
#   (nh=8, blocks=8, no diagram encoder, linear embed). amp_orders derives to [0,0]
#   for ee_uug (length-2, matches the base's n_order_features=2). Held-out NEAR region
#   (uug_heldtest.npz) is NEVER in any of these training sets -> a clean extrapolation
#   test at f=0, in-support baseline at f=1.
python run.py \
  exp_name=pretrain22_heldout_uug run_name=ft_f${TAG} \
  data.source=files data.data_path="$DATA_PATH" \
  'data.dataset=[ee_uug_91-1000GeV_amplitudes]' \
  data.preprocess_per_dataset=true \
  'data.train_test_val=[0.9, 0.05, 0.05]' \
  data.subsample=null \
  model.use_diagrams=false model.particle_encoder_hidden=0 \
  model.net.num_blocks=8 model.net.num_heads=8 \
  fine_tune.pretrained_path="$CKPT" \
  fine_tune.lr_scale=0.339 fine_tune.layer_decay=0.999 \
  training.lr=0.004 training.iterations=4000 training.batchsize=16384 \
  training.regularization=L2 training.regularization_lambda=2.47e-7 \
  training.scheduler=CosineAnnealingLR training.loss_aggregation=geometric_mean \
  training.cosanneal_warmup_frac=0.191 training.cosanneal_eta_min=1.6e-7 \
  plot=true save=true

echo "DONE ft_f${TAG}"
