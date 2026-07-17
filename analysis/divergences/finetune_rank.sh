#!/bin/bash
#SBATCH --job-name=q2rank
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --array=0-3
#SBATCH --output=analysis/divergences/q2rank_%A_%a.out
#SBATCH --error=analysis/divergences/q2rank_%A_%a.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
REPO=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
CKPT=$REPO/runs/pretrain22_heldout_uug/base/models/model_run0_best.pt

# Q2 rank-vs-magnitude: same base22/HPs/200k budget/4000 steps as every other Q2 arm;
# only the training density differs. These pools are MAGNITUDE-CONTROLLED (same weight
# multiset Phi=sort|r| reassigned by different orderings) -> isolates rank from magnitude.
TAGS=(rank_real rank_synth30 rank_synth46 rank_synth70)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
DATA_PATH=$REPO/data_reweight_${TAG}/

python run.py \
  exp_name=q2_sigma_reweight run_name=ft_${TAG} \
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

echo "DONE ft_${TAG}"
