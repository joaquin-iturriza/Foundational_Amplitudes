#!/bin/bash
#SBATCH --job-name=bigrun_arm
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --account=lpnhe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=compare_models/bigrun_arm_%x_%j.out
#SBATCH --error=compare_models/bigrun_arm_%x_%j.err
# One feature-ablation arm at H* (baseline all-on minus one feature). $ARM_OVR passed via --export.
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes
export AMP_FROZEN_DIR=$SCRATCH/datasets_scanbig_cut
export AMP_TRAIN_CACHE_DIR=$SCRATCH/amp_cache_scanbig_cut
export AMP_FIDUCIAL_CUTS=on
python run.py model=lloca local=none \
  model.net.num_heads=8 model.net.num_blocks=8 \
  data.source=recipes data.processes_file=$PWD/recipes/scan_bigrun_100k.yaml \
  data.require_cache=true data.preprocess_per_dataset=true \
  data.train_subsample=null data.eval_subsample=2000 data.seed=42 seed=42 \
  data.use_PIDs=false data.spin_onehot=true data.color_onehot=true \
  data.prop_is_massless=true data.standardize_props=true \
  data.mass_from_momenta=true data.coupling_scalars=true \
  data.internal_mass_scalars=true data.offshell_per_event=true \
  data.internal_mass_pdgs='[23,6,25]' model.use_diagrams=true model.d_diag=32 \
  training.batchsize=16384 evaluation.batchsize=16384 \
  training.loss_aggregation=geometric_mean training.regularization=L2 \
  training.scheduler=CosineAnnealingLR training.dtype=float32 \
  training.get_ID=false use_mlflow=false \
  training.lr=4.079396e-03 training.regularization_lambda=7.001527e-07 \
  training.cosanneal_warmup_frac=1.494990e-01 training.cosanneal_eta_min=5.142869e-10 \
  ema=true training.ema_decay=9.881993e-01 training.iterations=8601 evaluate=false plot=true training.result_path=$PWD/compare_models/_bigrun_arm_${ARM}/result.json \
  $ARM_OVR \
  exp_name=bigrun_arm_${ARM} run_dir=$PWD/compare_models/_bigrun_arm_${ARM}
echo "===== ARM $ARM DONE ====="
