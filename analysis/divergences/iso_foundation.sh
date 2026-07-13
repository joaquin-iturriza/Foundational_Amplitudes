#!/bin/bash
#SBATCH --job-name=iso_found
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=analysis/divergences/iso_found_%A_%a.out
#SBATCH --error=analysis/divergences/iso_found_%A_%a.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
WT=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-heterosc
MAIN=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
cd "$WT"

# ISOLATION TEST: is the coupled finetune's residual mu gap (best ~7.4e-3 vs MSE 8.5e-5) due
# to the heteroscedastic LOSS/finetune-path, or to the HETEROSC FOUNDATION it warm-starts from?
# Both arms use loss=MSE and the EXACT ft_deep_antenna HPs (lr_scale 0.339, layer_decay 0.999,
# lr 4e-3, 4000 steps, antenna uug). The ONLY variable is the warm-start foundation.
#   arm 0 (ISO)  : HETEROSC foundation, mu-row-sliced 2ch->1ch (body AND mu-head preserved, so
#                  NO head-reset confound; see slice_mu_head.py)
#   arm 1 (CTRL) : MSE foundation (the baseline's own start) -> should reproduce ~8.5e-5 here,
#                  confirming the worktree code path and pinning the run-to-run noise scale.
# Read-out: arm0 ~8.5e-5  => foundation is FINE, residual is the heteroscedastic finetune path.
#           arm0 ~7e-3    => the HETEROSC FOUNDATION body is the bottleneck.
NAMES=(iso_mse_from_het ctrl_mse_from_mse)
CKPTS=(
  "$WT/runs/heterosc_foundation/het_lr0.004_b1.0/models/model_run0_best_mu1ch.pt"
  "$MAIN/runs/pretrain22_heldout_uug/base/models/model_run0_best.pt"
)
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
CKPT=${CKPTS[$SLURM_ARRAY_TASK_ID]}
DATA_PATH=$MAIN/data_deep_antenna/

python run.py \
  exp_name=heterosc_iso run_name=${NAME} \
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

echo "DONE ${NAME}"
