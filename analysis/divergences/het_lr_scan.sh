#!/bin/bash
#SBATCH --job-name=het_lr
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=analysis/divergences/hetlr_%A_%a.out
#SBATCH --error=analysis/divergences/hetlr_%A_%a.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
WT=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-heterosc
MAIN=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
cd "$WT"

# FROM-SCRATCH A/B: does HETEROSC match MSE at equal budget in THIS codebase?
# Direct test of the claim that het loss works fine for amplitude regression (same accuracy,
# calibrated sigma). NO warm start, NO foundation -- fresh init both sides, identical budget,
# identical HPs, same data. The only difference is the loss.
#   arm 0 mse     : loss=MSE       (reference)
#   arm 1 het_b1  : loss=HETEROSC, beta=1
# If het ~= mse here  -> the het implementation is FINE, and the failure is specific to the
#                        warm-start/finetune path (foundation sigma head mismatched to the
#                        new error scale) -> that is where the bug is.
# If het >> mse here  -> the het implementation itself is broken, independent of finetuning.
# FAIR lr SCAN for HETEROSC (beta=1), FROM SCRATCH, same budget/data as scratch_mse (9.19e-5).
# het lost 1000x at MSE's HPs (lr=4e-3). Per the A/B protocol the feature now gets its OWN HPs;
# lr is the knob flagged as most likely. If het's best-over-lr reaches ~1e-4 -> the implementation
# is fine and it was purely an lr mismatch. If it plateaus ~0.09 at EVERY lr -> the implementation
# is broken, independent of HPs.
LRS=(1e-4 4e-4 1e-3 4e-3 1e-2 3e-2)
i=$SLURM_ARRAY_TASK_ID
LR=${LRS[$i]}
NAME=het_lr${LR}
LOSS=HETEROSC
EXTRA="training.heterosc_beta=1.0 model.net.detach_sigma_backbone=false plotting.plot_mse_het=false"
DATA_PATH=$MAIN/data_deep_antenna/

python run.py \
  exp_name=heterosc_lrscan run_name=${NAME} \
  data.source=files data.data_path="$DATA_PATH" \
  'data.dataset=[ee_uug_91-1000GeV_amplitudes]' \
  data.preprocess_per_dataset=true \
  'data.train_test_val=[0.9, 0.05, 0.05]' \
  data.subsample=null \
  model.use_diagrams=false model.particle_encoder_hidden=0 \
  model.net.num_blocks=8 model.net.num_heads=8 \
  training.loss=${LOSS} \
  training.lr=${LR} training.iterations=4000 training.batchsize=16384 \
  training.regularization=L2 training.regularization_lambda=2.47e-7 \
  training.scheduler=CosineAnnealingLR \
  training.cosanneal_warmup_frac=0.191 training.cosanneal_eta_min=1.6e-7 \
  training.clip_grad_norm=5 \
  ${EXTRA} \
  plot=true save=true

echo "DONE ${NAME}"
