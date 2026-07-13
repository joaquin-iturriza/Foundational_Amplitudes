#!/bin/bash
#SBATCH --job-name=ft_hetv2
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=analysis/divergences/ft_hetv2_%j.out
#SBATCH --error=analysis/divergences/ft_hetv2_%j.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
WT=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-heterosc
MAIN=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
cd "$WT"

# Finetune the BEST heteroscedastic foundation (lr4e-3, beta=1; mu-MSE 0.025) to the
# held-out uug on the dense-IR antenna pool. Both sides are HETEROSC (2-ch head) so NO
# head reset — body AND (mu,sigma) head transfer. beta=1 + arithmetic-mean agg (auto).
# Goal: a WELL-FIT mu, then redo the sigma(x) map to see clean epistemic sigma.
CKPT=$WT/runs/heterosc_foundation/het_lr0.004_b1.0/models/model_run0_best.pt
DATA_PATH=$MAIN/data_deep_antenna/

python run.py \
  exp_name=heterosc_foundation run_name=ft_uug_het \
  data.source=files data.data_path="$DATA_PATH" \
  'data.dataset=[ee_uug_91-1000GeV_amplitudes]' \
  data.preprocess_per_dataset=true \
  'data.train_test_val=[0.9, 0.05, 0.05]' \
  data.subsample=null \
  model.use_diagrams=false model.particle_encoder_hidden=0 \
  model.net.num_blocks=8 model.net.num_heads=8 \
  fine_tune.pretrained_path="$CKPT" \
  fine_tune.lr_scale=0.339 fine_tune.layer_decay=0.999 \
  training.loss=HETEROSC training.heterosc_beta=1.0 \
  training.lr=0.004 training.iterations=4000 training.batchsize=16384 \
  training.regularization=L2 training.regularization_lambda=2.47e-7 \
  training.scheduler=CosineAnnealingLR \
  training.cosanneal_warmup_frac=0.191 training.cosanneal_eta_min=1.6e-7 \
  plot=true save=true

echo "DONE ft_uug_het"
