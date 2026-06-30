#!/bin/bash
#SBATCH --job-name=ab_oob
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=compare_models/ab_oob_%j.out
#SBATCH --error=compare_models/ab_oob_%j.err
#
# Reproduce the scalar-arm CUDA device-side assert (ScatterGatherKernel index OOB)
# under CUDA_LAUNCH_BLOCKING=1 so the traceback points at the REAL bad kernel
# (not the downstream torch.stack().tolist() where it currently surfaces).
# Uses the exact HPs of failed trial hp0154 + the scalar feature flags.

module load anaconda-py3/2023.09
source /gpfslocalsup/pub/anaconda-py3/2023.09/etc/profile.d/conda.sh
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
PROJ=$PWD

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
RUNDIR="$PROJ/compare_models/_ab_oob/hp0154"; rm -rf "$RUNDIR"
echo "### OOB repro: scalar(mass+coupling), hp0154 HPs, CUDA_LAUNCH_BLOCKING=1 ###"
python run.py model=lloca local=none \
  data.source=recipes "data.processes_file=${PROJ}/recipes/scan_bigrun.yaml" \
  data.require_cache=true data.preprocess_per_dataset=true \
  data.train_subsample=2000 data.eval_subsample=500 data.seed=42 seed=42 \
  data.use_PIDs=false data.spin_onehot=true data.color_onehot=true \
  data.prop_is_massless=true data.standardize_props=true \
  data.mass_from_momenta=true data.coupling_scalars=true model.use_diagrams=false \
  training.batchsize=1024 evaluation.batchsize=4096 \
  training.loss_aggregation=geometric_mean training.regularization=L2 \
  training.scheduler=CosineAnnealingLR \
  training.lr=0.008519829214306301 \
  training.regularization_lambda=7.81029041312059e-07 \
  training.cosanneal_warmup_frac=0.008222314156591892 \
  training.cosanneal_eta_min=1.4380570775024342e-08 \
  training.ema_decay=0.9725959913064726 \
  training.iterations=1500 training.validate_frac=0.34 \
  training.get_ID=false training.dtype=float32 use_mlflow=false plot=false \
  exp_name="ab_oob_hp0154" run_dir="$RUNDIR" 2>&1 \
  | grep -vE "Skipping update|means=\[" | tail -60
echo ">>> repro done"
