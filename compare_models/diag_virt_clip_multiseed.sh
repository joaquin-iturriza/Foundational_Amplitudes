#!/bin/bash
#SBATCH --job-name=dvclipms
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --array=0-4%5
#SBATCH --output=compare_models/dvclipms_%A_%a.out
#SBATCH --error=compare_models/dvclipms_%A_%a.err
#
# Diagram-conditioning A/B (cheap-shortcut, CLAUDE.md A/B protocol step 3):
# re-run the BEST encoding baseline ("combo": spin one-hot + massless flag + std)
# WITH Feynman-diagram conditioning ON, at combo's BO-best HPs, across the same 5
# seeds and the same 25-process recipe. The no-diagram baseline already exists
# (compare_models/multiseed25/combo_seed*.json) and is NOT rerun. Compare
# best-vs-best on val_loss_no_reg via aggregate_diag.py.

set -e
module load anaconda-py3/2023.09
source /gpfslocalsup/pub/anaconda-py3/2023.09/etc/profile.d/conda.sh
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
PROJ=$PWD

SEED=$SLURM_ARRAY_TASK_ID

# Same encoding as the "combo" baseline (so the ONLY difference is diagrams on/off)
ENC=(data.spin_onehot=true data.prop_is_massless=true data.standardize_props=true)

# combo's BO-best HP (identical to the baseline runs)
read LR REG WARM ETA EMA < <(python -c "import json;h=json.load(open('compare_models/encab25_best_hps.json'))['combo']['hp'];print(h['training.lr'],h['training.regularization_lambda'],h['training.cosanneal_warmup_frac'],h['training.cosanneal_eta_min'],h['training.ema_decay'])")

mkdir -p compare_models/multiseed25
RES="$PROJ/compare_models/multiseed25/diag_virt_clip_seed${SEED}.json"
echo "=== diag seed=$SEED  lr=$LR reg=$REG ema=$EMA  $(date +%H:%M:%S) ==="
rm -rf "$PROJ/runs/dvclipms/s${SEED}"
python run.py model=lloca local=none \
  data.source=recipes "data.processes_file=${PROJ}/recipes/pretrain25_short.yaml" "data.data_path=${PROJ}/data/" data.require_cache=true \
  data.preprocess_per_dataset=true data.subsample=null "${ENC[@]}" \
  model.use_diagrams=true model.use_diagram_virtuality=true model.d_diag=32 \
  seed=$SEED training.batchsize=1024 evaluation.batchsize=8192 \
  training.iterations=4500 training.validate_every_n_steps=225 training.validate_frac=0.05 \
  training.loss_aggregation=geometric_mean training.regularization=L2 \
  training.lr=$LR training.clip_grad_norm=1.0 training.regularization_lambda=$REG \
  training.cosanneal_warmup_frac=$WARM training.cosanneal_eta_min=$ETA training.ema_decay=$EMA \
  training.scheduler=CosineAnnealingLR training.save_intermediate=false \
  training.get_ID=false training.dtype=float32 use_mlflow=false plot=false \
  training.result_path="$RES" \
  exp_name="dvclipms_s${SEED}" run_dir="$PROJ/runs/dvclipms/s${SEED}"
