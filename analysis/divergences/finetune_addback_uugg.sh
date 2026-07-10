#!/bin/bash
#SBATCH --job-name=ft_uugg_ho
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=0-4
#SBATCH --output=analysis/divergences/ft_uugg_ho_%A_%a.out
#SBATCH --error=analysis/divergences/ft_uugg_ho_%A_%a.out

set -e
module purge
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

REPO=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
CKPT=$REPO/runs/pretrain22_heldout_uug/base/models/model_run0_best.pt

# ee->uugg hold-out/add-back. base22 held out ALL five gluon processes (incl. uugg),
# so this reuses the SAME foundation as the uug study - no new pretrain. uugg is the
# hardest process (6-particle) and the sole "degradation into a singularity" case, so
# the question is whether its divergence is also data-recoverable. Same y_min<c cut,
# same finetune HPs as finetune_addback.sh. f=0 pure extrapolation, f=1 in-support.
TAGS=(000 005 015 050 100)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
DATA_PATH=$REPO/data_ho_uugg_f${TAG}/

python run.py \
  exp_name=pretrain22_heldout_uug run_name=ft_uugg_f${TAG} \
  data.source=files data.data_path="$DATA_PATH" \
  'data.dataset=[ee_uugg_91-1000GeV_amplitudes]' \
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

echo "DONE ft_uugg_f${TAG}"
