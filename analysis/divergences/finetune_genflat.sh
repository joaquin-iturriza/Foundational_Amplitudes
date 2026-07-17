#!/bin/bash
#SBATCH --job-name=ft_genflat
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=analysis/divergences/ft_genflat_%j.out
#SBATCH --error=analysis/divergences/ft_genflat_%j.out

set -e
module purge
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

REPO=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
CKPT=$REPO/runs/pretrain22_heldout_uug/base/models/model_run0_best.pt

# L0 done PROPERLY: GENERATED training data, not resampled. data_genflat_eeuu/ holds fresh
# ee->uu events importance-sampled toward flat log|M|^2 (gen_uu_sampling.py: pole-covering
# sqrt(s) proposal -> exact-|M|^2 labels -> thin to flat per amplitude decade WITHOUT
# replacement, so every event is unique). This breaks the coverage ceiling that doomed the
# resampled pool: the Z pole now has ~30% of events as GENUINE new points (vs 0.4% RAMBO).
# Baseline = the existing ft_raw arm (native RAMBO subsample), REUSED per A/B rule #1. Same
# base8 checkpoint, same HPs, same N=400k, same steps -- only the training density differs.
# Eval bins the common held-out RAMBO test by sqrt(s) (never used by the generator's shaper).
DATA_PATH=$REPO/data_genflat_eeuu/

python run.py \
  exp_name=eeuu_flatlogm run_name=ft_genflat \
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

echo "DONE ft_genflat"
