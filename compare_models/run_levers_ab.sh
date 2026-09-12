#!/bin/bash
#SBATCH --job-name=levab
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --account=lpnhe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:55:00
#SBATCH --array=0-1
#SBATCH --output=compare_models/levab_%A_%a.out
#SBATCH --error=compare_models/levab_%A_%a.err
#
# Focused off-vs-offshell A/B on the internal-mass levers only (44 datasets: ee_mumu
# Z-scan, ee_mumumumu Z-4l, ee_wwbb top, ee_mumutautau Higgs), FULL train data (no
# subsample) and long training — so the comparison is in a well-trained regime, not
# undertraining noise. If the offshell arm flattens the mass-scan U-shape that off
# can't resolve, the internal-mass off-shellness feature works (generalising reson to
# top/Higgs/Z-4l). Per-dataset preprocessing; cut-tagged cache.

source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes
PROJ=$PWD
export AMP_TRAIN_CACHE_DIR=$SCRATCH/amp_cache_scanbig_cut
export AMP_FROZEN_DIR=$SCRATCH/datasets_scanbig_cut
export AMP_FIDUCIAL_CUTS=on

if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
  N=off;      MFM=false; CPL=false; IMS=false; OSH=false
else
  N=offshell; MFM=true;  CPL=true;  IMS=true;  OSH=true
fi
RUNDIR="$PROJ/compare_models/_levers_ab/$N"; rm -rf "$RUNDIR"
echo "### levers A/B $N (mass_from_momenta=$MFM coupling=$CPL internal_mass=$IMS offshell=$OSH) ###"
python run.py model=lloca local=none \
  data.source=recipes "data.processes_file=${PROJ}/recipes/scan_levers.yaml" \
  data.require_cache=true data.preprocess_per_dataset=true \
  data.train_subsample=null data.eval_subsample=2000 data.seed=42 seed=42 \
  data.use_PIDs=false data.spin_onehot=true data.color_onehot=true \
  data.prop_is_massless=true data.standardize_props=true \
  data.mass_from_momenta=$MFM data.coupling_scalars=$CPL \
  data.internal_mass_scalars=$IMS data.offshell_per_event=$OSH \
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
  exp_name="levab_$N" run_dir="$RUNDIR" \
  && echo ">>> $N done; val + plots in $RUNDIR/plots_0" || echo ">>> $N FAILED"
