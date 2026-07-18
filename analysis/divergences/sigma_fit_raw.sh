#!/bin/bash
#SBATCH --job-name=sigfit_eeuu
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=analysis/divergences/sigfit_eeuu_%j.out
#SBATCH --error=analysis/divergences/sigfit_eeuu_%j.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
WT=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-heterosc
MAIN=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
cd "$WT"

# L1 step 1 for ee->uu: fit a calibrated sigma head ON TOP OF the L0 coverage base (the f=0.25
# generated mixture, the sweep optimum), via the resolved two-stage recipe (freeze trunk+mu, train
# ONLY the sigma readout row at beta=0 -- a proper scoring rule with mu fixed; never train mu and
# sigma jointly). Stage-1 mu = the raw arm (runs/eeuu_flatlogm/ft_raw), so sigma learns the
# residual structure of a model that ALREADY has balanced pole+bulk coverage -- L1 then adds
# emphasis where sigma is large. GATE: requires ft_mix025/models/.

STAGE1=$MAIN/runs/eeuu_flatlogm/ft_raw
MU_CKPT=$STAGE1/models/model_run0_best.pt.gz     # cleanup gzips the ckpt after training
GROWN=$STAGE1/models/model_run0_best_2ch.pt
DATA_PATH=$MAIN/data_raw_eeuu/

# grow the sigma row on the converged mu head; seed sigma at the raw(RAMBO) model's own RMS residual
python analysis/divergences/grow_sigma_head.py "$MU_CKPT" "$GROWN" 1e-2

python run.py \
  exp_name=eeuu_sigfit run_name=raw_sigma \
  data.source=files data.data_path="$DATA_PATH" \
  'data.dataset=[ee_uu_91-1000GeV_amplitudes]' \
  data.preprocess_per_dataset=true \
  'data.train_test_val=[0.9, 0.05, 0.05]' \
  data.subsample=null \
  model.use_diagrams=false model.particle_encoder_hidden=0 \
  model.net.num_blocks=8 model.net.num_heads=8 \
  model.net.detach_sigma_backbone=false \
  model.net.sigma_after_pool=true \
  fine_tune.pretrained_path="$GROWN" \
  fine_tune.reset_output_head=false \
  fine_tune.lr_scale=1.0 fine_tune.layer_decay=1.0 \
  training.loss=HETEROSC training.heterosc_beta=0.0 training.heterosc_sigma_only=true \
  training.lr=0.0202319 training.iterations=2000 training.batchsize=16384 \
  evaluation.batchsize=16384 training.clip_grad_norm=5 \
  training.regularization=L2 training.regularization_lambda=9.892346e-07 \
  training.scheduler=CosineAnnealingLR \
  training.cosanneal_warmup_frac=0.1537129 training.cosanneal_eta_min=1.0e-8 \
  plot=true save=true

echo "DONE sigfit_eeuu (mu frozen -> mu-MSE must equal the L0 flat arm; check sigma calibration)"
