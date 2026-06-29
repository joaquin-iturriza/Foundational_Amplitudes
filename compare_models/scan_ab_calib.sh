#!/bin/bash
#SBATCH --job-name=scan_ab_calib
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=compare_models/scan_ab_calib_%j.out
#SBATCH --error=compare_models/scan_ab_calib_%j.err
#
# Calibration for the 3-way coupling+mass A/B on the full 385-dataset scan set:
# one BASELINE run (all features off) over the prebuilt cache to (a) confirm the
# full dataset trains end-to-end with require_cache=true, (b) measure per-step time
# so the A/B sweep can be sized. LLOCA_PROFILE_STEP prints a data-vs-compute split.

module load anaconda-py3/2023.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
PROJ=$PWD
export LLOCA_PROFILE_STEP=1

python run.py \
  model=lloca local=none \
  data.source=recipes "data.processes_file=${PROJ}/recipes/scan_bigrun.yaml" \
  data.require_cache=true data.preprocess_per_dataset=true \
  data.train_subsample=2000 data.eval_subsample=1000 \
  data.seed=42 seed=42 \
  data.use_PIDs=false data.spin_onehot=true data.color_onehot=true \
  data.prop_is_massless=true data.standardize_props=true \
  data.mass_from_momenta=false data.coupling_scalars=false model.use_diagrams=false \
  training.batchsize=1024 evaluation.batchsize=4096 \
  training.loss_aggregation=geometric_mean \
  training.regularization=L2 training.regularization_lambda=1e-8 \
  training.scheduler=CosineAnnealingLR training.lr=2e-3 \
  training.iterations=2000 training.validate_frac=0.25 \
  training.get_ID=false training.dtype=float32 use_mlflow=false plot=true \
  exp_name=scan_ab_calib run_dir="$PROJ/compare_models/_scan_ab_calib" \
  && echo ">>> calib exited 0" || echo ">>> calib FAILED"
