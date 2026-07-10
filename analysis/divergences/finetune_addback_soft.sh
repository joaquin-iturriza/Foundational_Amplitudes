#!/bin/bash
#SBATCH --job-name=ft_soft_ho
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=analysis/divergences/ft_soft_ho_%A_%a.out
#SBATCH --error=analysis/divergences/ft_soft_ho_%A_%a.out

set -e
module purge
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

REPO=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
CKPT=$REPO/runs/pretrain22_heldout_uug/base/models/model_run0_best.pt

# ee->uug SOFT-region hold-out (x_g<c): pure-soft extrapolation, complementary to the
# collinear cut. Same base22 + finetune HPs as finetune_addback.sh. The soft-vs-collinear
# f=0 comparison answers WHICH IR limit is harder to extrapolate into.
TAGS=(000 005 015 100)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
DATA_PATH=$REPO/data_ho_soft_f${TAG}/

python run.py \
  exp_name=pretrain22_heldout_uug run_name=ft_soft_f${TAG} \
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

echo "DONE ft_soft_f${TAG}"
