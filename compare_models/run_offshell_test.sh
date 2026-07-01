#!/bin/bash
#SBATCH --job-name=offsh
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --array=0-2
#SBATCH --output=compare_models/offsh_%A_%a.out
#SBATCH --error=compare_models/offsh_%A_%a.err
#
# Production internal-mass conditioning: propagator OFF-SHELLNESS s_prop − M² fed
# DIRECTLY to the main transformer (data.offshell_per_event), the general form of the
# proven s-channel reson result (√s − M). ee_mumu across the Z peak, M_Z scanned.
#   arm 0 off     : no internal-mass feature            -> expect the U-shape
#   arm 1 reson   : s-channel √s−M direct (proven 0.00) -> the target to match
#   arm 2 offshell: general s−M² via diagram masks, DIRECT (no encoder), use_diagrams=false
# If arm 2 flattens the U like arm 1, the general per-propagator off-shellness works.
# Per-dataset standardization.

module load anaconda-py3/2023.09
source /gpfslocalsup/pub/anaconda-py3/2023.09/etc/profile.d/conda.sh
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
PROJ=$PWD
export AMP_TRAIN_CACHE_DIR=$SCRATCH/amp_cache_imz
export AMP_FROZEN_DIR=$SCRATCH/datasets_imz

case "$SLURM_ARRAY_TASK_ID" in
  0) N=off;      IMS=false; RPE=false; OSH=false ;;
  1) N=reson;    IMS=true;  RPE=true;  OSH=false ;;
  2) N=offshell; IMS=true;  RPE=false; OSH=true  ;;
esac
RUNDIR="$PROJ/compare_models/_offshell_test/$N"; rm -rf "$RUNDIR"
echo "### offshell test $N (internal_mass_scalars=$IMS resonance=$RPE offshell=$OSH pdg=23) ###"
python run.py model=lloca local=none \
  data.source=recipes "data.processes_file=${PROJ}/recipes/internal_mass_zpeak_test.yaml" \
  data.require_cache=true data.preprocess_per_dataset=true \
  data.eval_subsample=1000 data.seed=42 seed=42 \
  data.use_PIDs=false data.spin_onehot=true data.color_onehot=true \
  data.prop_is_massless=true data.standardize_props=true \
  data.mass_from_momenta=false data.coupling_scalars=false \
  data.internal_mass_scalars=$IMS data.resonance_per_event=$RPE data.offshell_per_event=$OSH \
  "data.internal_mass_pdgs=[23]" \
  model.use_diagrams=false \
  training.batchsize=1024 evaluation.batchsize=4096 \
  training.loss_aggregation=geometric_mean training.regularization=L2 \
  training.scheduler=CosineAnnealingLR \
  training.lr=0.0037850206125016168 training.regularization_lambda=1.4383791066176087e-10 \
  training.cosanneal_warmup_frac=0.037281612306833266 \
  training.cosanneal_eta_min=5.00942594074264e-10 training.ema_decay=0.9687763301244937 \
  training.iterations=8000 training.validate_frac=0.02 \
  training.get_ID=false training.dtype=float32 use_mlflow=false plot=true \
  exp_name="offsh_$N" run_dir="$RUNDIR" \
  && echo ">>> $N done; val + plots in $RUNDIR/plots_0" || echo ">>> $N FAILED"
