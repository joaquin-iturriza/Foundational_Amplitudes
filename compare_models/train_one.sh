#!/bin/bash
#SBATCH --job-name=amp_cmp
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=compare_models/cmp_%x_%j.out
#SBATCH --error=compare_models/cmp_%x_%j.err
#
# Stage B: one comparison run. Args: $1 = model key {lloca|lgatr|slim}, $2 = iterations.
# All three share an IDENTICAL recipe + HPs (25-process pretrain set, batch=1024,
# geometric-mean loss, same lr/reg/warmup/ema); only the architecture and the step
# count differ — step counts are set by Stage A so total training FLOPs match.
# Submit (after profiling) e.g.:  sbatch -J cmp_lloca compare_models/train_one.sh lloca 15000

set -e
MODEL=$1
ITERS=$2

module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
PROJ=$PWD

case "$MODEL" in
  lloca) MODEL_OV=(model=lloca model.net.num_blocks=8 model.net.num_heads=8) ;;
  lgatr) MODEL_OV=(model=lgatr_mup model.net.num_blocks=8
                   model.net.hidden_mv_channels=22 model.net.hidden_s_channels=22) ;;
  slim)  MODEL_OV=(model=lgatr_slim model.net.num_blocks=8
                   model.net.hidden_v_channels=52 model.net.hidden_s_channels=104) ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac

RUNDIR="$PROJ/runs/cmp_${MODEL}"
rm -rf "$RUNDIR"

# Shared recipe (mirrors sweep_config_jeanzay_pretrain25 fixed_params) + fixed HPs.
SHARED=(
  local=none
  data.source=recipes
  "data.processes_file=${PROJ}/recipes/pretrain25.yaml"
  "data.data_path=${PROJ}/data/"
  data.preprocess_per_dataset=true
  data.require_cache=true
  data.train_subsample=500000
  data.eval_subsample=20000
  data.seed=42
  seed=42
  training.batchsize=1024
  evaluation.batchsize=8192
  training.loss_aggregation=geometric_mean
  training.regularization=L2
  training.regularization_lambda=1e-8
  training.scheduler=CosineAnnealingLR
  training.lr=2e-3
  training.cosanneal_warmup_frac=0.05
  training.cosanneal_eta_min=1e-8
  training.ema_decay=0.999
  training.validate_frac=0.02
  training.save_intermediate=true
  training.get_ID=false
  training.dtype=float32
  plot=true
)

echo "=== $MODEL : iterations=$ITERS ==="
python run.py "${SHARED[@]}" "${MODEL_OV[@]}" \
  training.iterations="$ITERS" \
  exp_name="cmp_${MODEL}" \
  run_dir="$RUNDIR"
