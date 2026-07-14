#!/bin/bash
#SBATCH --job-name=data_ctrl
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=analysis/divergences/datactrl_%A_%a.out
#SBATCH --error=analysis/divergences/datactrl_%A_%a.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
WT=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-heterosc
MAIN=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
cd "$WT"

# DATA CONTROL. Everything about the reference's validated het setup differs from ours on TWO
# axes at once: architecture (MLP, per-event readout) and DATA (its TYPE_TOKEN_DICT is keyed on
# zg/zgg/aag/wz... — LHC Z/W+jets, flat-sampled — not ee_uug). Our runs are LLoCa on a
# deliberately IR-ENHANCED antenna pool that packs events into the soft/collinear poles
# (log|M|^2 std 3.93 vs the flat set's 2.25).
#
# This changes ONLY the data: same LLoCa model, same HETEROSC loss (beta=0 = plain NLL = the
# reference's), same HPs/budget, flat-RAMBO uug subsampled to 400k to match the antenna pool's N.
#   arm 0 flat_mse : accuracy reference on the flat data
#   arm 1 flat_het : het on the flat data
# Read-out: slope ~1.0 on flat (with mu actually fit) => the implementation is FINE and the
#   miscalibration is specific to the IR-enhanced distribution — a real property of the antenna
#   data, not a bug. slope ~1.4 on flat TOO => the pathology is not the data either.
# GUARD: slope~1 only counts if mu is fit -- an undertrained model gets slope~1 for free
# (measured: trial_0037 had slope 1.01 at mu-MSE 0.0265). Hence the MSE arm.
NAMES=(flat_mse flat_het)
LOSSES=(MSE     HETEROSC)
i=$SLURM_ARRAY_TASK_ID
NAME=${NAMES[$i]}; LOSS=${LOSSES[$i]}

EXTRA=""
[ "$LOSS" = "HETEROSC" ] && EXTRA="training.heterosc_beta=0.0 model.net.sigma_after_pool=true"
[ "$LOSS" = "MSE" ] && EXTRA="training.loss_aggregation=geometric_mean"

python run.py \
  exp_name=heterosc_datactrl run_name=${NAME} \
  data.source=files data.data_path="$MAIN/data/" \
  'data.dataset=[ee_uug_91-1000GeV_amplitudes]' \
  data.preprocess_per_dataset=true \
  'data.train_test_val=[0.9, 0.05, 0.05]' \
  data.subsample=400000 \
  model.use_diagrams=false model.particle_encoder_hidden=0 \
  model.net.num_blocks=8 model.net.num_heads=8 \
  training.loss=${LOSS} \
  training.lr=0.004 training.iterations=4000 training.batchsize=16384 \
  training.regularization=L2 training.regularization_lambda=2.47e-7 \
  training.scheduler=CosineAnnealingLR \
  training.cosanneal_warmup_frac=0.191 training.cosanneal_eta_min=1.6e-7 \
  training.clip_grad_norm=5 \
  ${EXTRA} \
  plot=true save=true

echo "DONE ${NAME}"
