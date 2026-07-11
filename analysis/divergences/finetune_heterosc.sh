#!/bin/bash
#SBATCH --job-name=ft_het
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=analysis/divergences/ft_het_%j.out
#SBATCH --error=analysis/divergences/ft_het_%j.out

set -e
module purge
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
# RUN FROM THE WORKTREE so the HETEROSC-wired LLoCa code is used.
WT=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-heterosc
MAIN=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
cd "$WT"

CKPT=$MAIN/runs/pretrain22_heldout_uug/base/models/model_run0_best.pt   # gz-robust loader
DATA_PATH=$MAIN/data_deep_antenna/    # dense-IR antenna pool (reaches y_min~1e-6)

# HETEROSC finetune of leave-uug-out base22 on the antenna pool. Same HPs as the MSE
# add-back finetunes; only the loss (Gaussian NLL, model emits mean+sigma) differs.
# reset_output_head=true: base22's 1-ch readout can't load into the new 2-ch head, so
# the body transfers and the (mu,sigma) head trains fresh. Goal: inspect sigma(x) across
# y_min — does it track model error now (epistemic-usable), or collapse to the float32 floor.
python run.py \
  exp_name=pretrain22_heldout_uug run_name=ft_deep_antenna_het \
  data.source=files data.data_path="$DATA_PATH" \
  'data.dataset=[ee_uug_91-1000GeV_amplitudes]' \
  data.preprocess_per_dataset=true \
  'data.train_test_val=[0.9, 0.05, 0.05]' \
  data.subsample=null \
  model.use_diagrams=false model.particle_encoder_hidden=0 \
  model.net.num_blocks=8 model.net.num_heads=8 \
  fine_tune.pretrained_path="$CKPT" fine_tune.reset_output_head=true \
  fine_tune.lr_scale=0.339 fine_tune.layer_decay=0.999 \
  training.loss=HETEROSC \
  training.lr=0.004 training.iterations=4000 training.batchsize=16384 \
  training.regularization=L2 training.regularization_lambda=2.47e-7 \
  training.scheduler=CosineAnnealingLR training.loss_aggregation=geometric_mean \
  training.cosanneal_warmup_frac=0.191 training.cosanneal_eta_min=1.6e-7 \
  plot=true save=true

echo "DONE ft_deep_antenna_het"
