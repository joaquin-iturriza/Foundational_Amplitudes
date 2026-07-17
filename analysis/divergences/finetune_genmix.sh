#!/bin/bash
#SBATCH --job-name=ft_genmix
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=analysis/divergences/ft_genmix_%A_%a.out
#SBATCH --error=analysis/divergences/ft_genmix_%A_%a.out

set -e
module purge
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

REPO=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
CKPT=$REPO/runs/pretrain22_heldout_uug/base/models/model_run0_best.pt

# Mixture fraction sweep (GENERATED): f = fraction of events drawn flat-log|M|^2 (pole-covering,
# fresh) vs (1-f) uniform-sqrt(s) RAMBO. Endpoints reused: f=0 == ft_raw, f=1 == ft_genflat.
# This array does the 3 interior cells. Same base8 ckpt, same HPs, same N=400k -- only the
# training density's mixture fraction changes. Locates the interior optimum that keeps the bulk
# sharp AND covers the pole (the uug fraction-sweep lesson, for an s-channel resonance).
TAGS=(mix025 mix050 mix075)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
DATA_PATH=$REPO/data_gen${TAG}_eeuu/

python run.py \
  exp_name=eeuu_flatlogm run_name=ft_${TAG} \
  data.source=files data.data_path="$DATA_PATH" \
  'data.dataset=[ee_uu_91-1000GeV_amplitudes]' \
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

echo "DONE ft_${TAG}"
