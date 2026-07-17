#!/bin/bash
#SBATCH --job-name=ft_flatlogm
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=analysis/divergences/ft_flatlogm_%A_%a.out
#SBATCH --error=analysis/divergences/ft_flatlogm_%A_%a.out

set -e
module purge
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

REPO=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
CKPT=$REPO/runs/pretrain22_heldout_uug/base/models/model_run0_best.pt

# L0 validation A/B (Thread D general default sampling): same foundation checkpoint, same
# #events (400k), IDENTICAL HPs — the ONLY difference is the phase-space sampling density of
# the ee->uu training pool:
#   TAG=raw      : native RAMBO density        -> s-channel Z resonance starved (0.4% of events)
#   TAG=flatlogm : flat in log|M|^2 (L0)       -> equal density per amplitude decade (resonance covered)
# The resampler (flat_logm_resample.py) reads ONLY the amplitude column; sqrt(s) is used ONLY
# at eval to prove the coverage fix is structure-agnostic. HPs identical to finetune_deep_sampling.sh
# (best deep cell). Per-dataset preprocessing standardizes each pool independently -> eval compares
# on de-standardized log|M|^2 over the COMMON held-out test (data_test_eeuu/).
TAGS=(raw flatlogm)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
DATA_PATH=$REPO/data_${TAG}_eeuu/

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
