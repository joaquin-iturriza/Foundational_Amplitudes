#!/bin/bash
#SBATCH --job-name=ft_flatlogm_sh
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=analysis/divergences/ft_flatlogm_sh_%A_%a.out
#SBATCH --error=analysis/divergences/ft_flatlogm_sh_%A_%a.out

set -e
module purge
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
WT=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-heterosc
MAIN=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
cd "$WT"                                    # run from the worktree: data.frozen_stats_path lives here

CKPT=$MAIN/runs/pretrain22_heldout_uug/base/models/model_run0_best.pt
STATS=$MAIN/analysis/divergences/eeuu_shared_stats.json   # native-pool stats, SHARED by both arms

# L0 validation A/B, FAIR version. The first pass was confounded: per-dataset standardization
# fit a DIFFERENT amp mean/std on each pool (raw std 0.97 vs flat 2.53), silently rescaling the
# log-amp loss ~6.8x and burying the coverage effect (CLAUDE.md A/B rule #6). Here BOTH arms load
# the SAME frozen stats (computed once on the native RAMBO pool), so the ONLY difference is the
# sampling density of the ee->uu training pool:
#   TAG=raw      : native RAMBO   -> Z resonance starved
#   TAG=flatlogm : flat log|M|^2  -> equal density per amplitude decade
TAGS=(raw flatlogm)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
DATA_PATH=$MAIN/data_${TAG}_eeuu/

python run.py \
  base_dir=$MAIN \
  exp_name=eeuu_flatlogm_shared run_name=ft_${TAG} \
  data.source=files data.data_path="$DATA_PATH" \
  'data.dataset=[ee_uu_91-1000GeV_amplitudes]' \
  data.preprocess_per_dataset=true \
  data.frozen_stats_path="$STATS" \
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

echo "DONE ft_${TAG} (shared stats)"
