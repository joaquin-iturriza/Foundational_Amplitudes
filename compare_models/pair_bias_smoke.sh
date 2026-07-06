#!/bin/bash
#SBATCH --job-name=pb_smoke
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=compare_models/pb_smoke_%j.out
#SBATCH --error=compare_models/pb_smoke_%j.err
#
# End-to-end smoke of the pairwise attention bias (model.use_pair_bias) on GPU:
# spec build from sidecars, setup_pair_bias, padded-SDPA attention path (replaces
# the xformers block-diagonal kernel when the bias is on), first-batch calibration
# buffers, train->validate->save->plot, on the tiny 8-process set, ~40 steps.
# Config = the adopted big-run candidate (no diagram encoder, linear embed) + bias.
# A bias-OFF twin guards the untouched default path through the same merged code.

module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
PROJ=$PWD

COMMON=(
  model=lloca
  local=none
  data.source=files
  "data.data_path=${PROJ}/data_small_encab/"
  data.preprocess_per_dataset=true
  data.subsample=null
  data.seed=42
  seed=42
  data.spin_onehot=true
  data.color_onehot=true
  data.prop_is_massless=true
  data.standardize_props=true
  # adopted big-run candidate: no diagram encoder, linear particle embed
  model.use_diagrams=false
  model.particle_encoder_hidden=0
  training.batchsize=1024
  evaluation.batchsize=4096
  training.loss_aggregation=geometric_mean
  training.regularization=L2
  training.regularization_lambda=1e-8
  training.scheduler=CosineAnnealingLR
  training.lr=2e-3
  training.iterations=40
  training.validate_frac=0.25
  training.save_intermediate=true
  training.get_ID=false
  training.dtype=float32
  use_mlflow=false
  plot=true
)

run_one () {
  local NAME=$1; shift
  local RUNDIR="$PROJ/compare_models/_pb_smoke_${NAME}"
  rm -rf "$RUNDIR"
  echo "############################## SMOKE $NAME ##############################"
  python run.py "${COMMON[@]}" "$@" exp_name="pb_smoke_${NAME}" run_dir="$RUNDIR" \
    && echo ">>> $NAME: python exited 0" || echo ">>> $NAME: python FAILED"
}

run_one pb_on  model.use_pair_bias=true
run_one pb_off model.use_pair_bias=false

echo "===== PB SMOKE DONE ====="
for N in pb_on pb_off; do
  D="$PROJ/compare_models/_pb_smoke_${N}"
  echo "[$N] models: $(ls $D/models 2>/dev/null | tr '\n' ' ')"
  echo "[$N] result: $(cat $D/result.json 2>/dev/null | head -c 300)"
done
