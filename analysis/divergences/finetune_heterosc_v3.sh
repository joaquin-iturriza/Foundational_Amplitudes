#!/bin/bash
#SBATCH --job-name=ft_hetv3
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=analysis/divergences/ft_hetv3_%j.out
#SBATCH --error=analysis/divergences/ft_hetv3_%j.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
WT=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-heterosc
MAIN=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
cd "$WT"

# v3: IDENTICAL to finetune_heterosc_v2.sh but with model.net.detach_sigma_backbone=true.
# Diagnosis (diag_ckpt_mu.py) ruled out checkpoint selection (best==last) and sigma-inflation
# (sigma calibrated, pull_std~0.93); the HETEROSC foundation mu-MSE (0.025) matches the MSE
# foundation (0.020), so beta=1 fits mu FINE at the foundation. Only the single-process uug
# finetune underfits (mu-MSE 0.086 vs MSE finetune 8.5e-5). Hypothesis: (mu,sigma) multi-task
# interference through the SHARED backbone floors mu where the mean is easy to fit. detach_
# sigma_backbone gives the backbone only mu's (pure-MSE) gradient. Expect mu-MSE ~8.5e-5 if
# the hypothesis holds; still-stuck -> it's finetune HPs, sweep next.
CKPT=$WT/runs/heterosc_foundation/het_lr0.004_b1.0/models/model_run0_best.pt
DATA_PATH=$MAIN/data_deep_antenna/

python run.py \
  exp_name=heterosc_foundation run_name=ft_uug_het_detach \
  data.source=files data.data_path="$DATA_PATH" \
  'data.dataset=[ee_uug_91-1000GeV_amplitudes]' \
  data.preprocess_per_dataset=true \
  'data.train_test_val=[0.9, 0.05, 0.05]' \
  data.subsample=null \
  model.use_diagrams=false model.particle_encoder_hidden=0 \
  model.net.num_blocks=8 model.net.num_heads=8 \
  model.net.detach_sigma_backbone=true \
  fine_tune.pretrained_path="$CKPT" \
  fine_tune.lr_scale=0.339 fine_tune.layer_decay=0.999 \
  training.loss=HETEROSC training.heterosc_beta=1.0 \
  training.lr=0.004 training.iterations=4000 training.batchsize=16384 \
  training.regularization=L2 training.regularization_lambda=2.47e-7 \
  training.scheduler=CosineAnnealingLR \
  training.cosanneal_warmup_frac=0.191 training.cosanneal_eta_min=1.6e-7 \
  plotting.plot_mse_het=false plot=true save=true

echo "DONE ft_uug_het_detach"
