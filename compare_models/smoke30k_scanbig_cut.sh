#!/bin/bash
#SBATCH --job-name=smoke30k
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --account=lpnhe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:55:00
#SBATCH --output=compare_models/smoke30k_%j.out
#SBATCH --error=compare_models/smoke30k_%j.err
#
# Quick smoke on the NEW cut big-run data: does it TRAIN, and does every dataset
# learn (per-dataset val loss) now that the IR/forward outliers are cut? offshell arm
# (Z/top/Higgs internal-mass features on) so the mass-scan datasets can actually
# resolve their mass. Frequent validation (every 500 steps) for readable curves.

source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes
PROJ=$PWD
export AMP_TRAIN_CACHE_DIR=$SCRATCH/amp_cache_scanbig_cut
export AMP_FROZEN_DIR=$SCRATCH/datasets_scanbig_cut
export AMP_FIDUCIAL_CUTS=on
RUNDIR="$PROJ/compare_models/_smoke30k_cut"; rm -rf "$RUNDIR"

python run.py model=lloca local=none \
  data.source=recipes "data.processes_file=${PROJ}/recipes/scan_bigrun.yaml" \
  data.require_cache=true data.preprocess_per_dataset=true \
  data.train_subsample=2000 data.eval_subsample=1000 data.seed=42 seed=42 \
  data.use_PIDs=false data.spin_onehot=true data.color_onehot=true \
  data.prop_is_massless=true data.standardize_props=true \
  data.mass_from_momenta=true data.coupling_scalars=true \
  data.internal_mass_scalars=true data.offshell_per_event=true \
  "data.internal_mass_pdgs=[23,6,25]" \
  model.use_diagrams=false \
  training.batchsize=1024 evaluation.batchsize=4096 \
  training.loss_aggregation=geometric_mean training.regularization=L2 \
  training.scheduler=CosineAnnealingLR \
  training.lr=0.0037850206125016168 training.regularization_lambda=1.4383791066176087e-10 \
  training.cosanneal_warmup_frac=0.037281612306833266 \
  training.cosanneal_eta_min=5.00942594074264e-10 training.ema_decay=0.9687763301244937 \
  training.iterations=30000 training.validate_every_n_steps=2000 \
  training.get_ID=false training.dtype=float32 use_mlflow=false plot=true \
  exp_name="smoke30k_cut" run_dir="$RUNDIR" \
  && echo ">>> smoke done; plots + per-dataset val in $RUNDIR/plots_0" || echo ">>> smoke FAILED"
