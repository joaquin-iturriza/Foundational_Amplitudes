#!/bin/bash
#SBATCH --job-name=dvbestms
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --array=0-4%5
#SBATCH --output=compare_models/dvbestms_%A_%a.out
#SBATCH --error=compare_models/dvbestms_%A_%a.err
# Stability test of the Tier B HPO-sweep OPTIMUM: re-run the sweep's single best
# config across 5 seeds. Tells us if the best Tier B point is reproducibly stable.
set -e
module load anaconda-py3/2023.09
source /gpfslocalsup/pub/anaconda-py3/2023.09/etc/profile.d/conda.sh
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
PROJ=$PWD
SEED=$SLURM_ARRAY_TASK_ID
read LR REG WARM ETA EMA SCALE < <(python -c "import json;h=json.load(open('compare_models/dvirt_best_hp.json'));print(h['training.lr'],h['training.regularization_lambda'],h['training.cosanneal_warmup_frac'],h['training.cosanneal_eta_min'],h['training.ema_decay'],h['model.virt_log_scale'])")
mkdir -p compare_models/multiseed25
rm -rf "$PROJ/runs/dvbestms/s${SEED}"
RES="$PROJ/compare_models/multiseed25/dvirt_best_seed${SEED}.json"
python run.py model=lloca local=none \
  data.source=recipes "data.processes_file=${PROJ}/recipes/pretrain25_short.yaml" "data.data_path=${PROJ}/data/" data.require_cache=true \
  data.preprocess_per_dataset=true data.subsample=null data.spin_onehot=true data.prop_is_massless=true data.standardize_props=true \
  model.use_diagrams=true model.use_diagram_virtuality=true model.virt_standardize=false model.d_diag=32 model.virt_log_scale=$SCALE \
  seed=$SEED training.batchsize=1024 evaluation.batchsize=8192 \
  training.iterations=4500 training.validate_every_n_steps=225 training.validate_frac=0.05 \
  training.loss_aggregation=geometric_mean training.regularization=L2 \
  training.lr=$LR training.regularization_lambda=$REG \
  training.cosanneal_warmup_frac=$WARM training.cosanneal_eta_min=$ETA training.ema_decay=$EMA \
  training.scheduler=CosineAnnealingLR training.save_intermediate=false \
  training.get_ID=false training.dtype=float32 use_mlflow=false plot=false \
  training.result_path="$RES" exp_name="dvbest_s${SEED}" run_dir="$PROJ/runs/dvbestms/s${SEED}"
