#!/bin/bash
#SBATCH --job-name=ab_dvsmoke
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=compare_models/ab_dvsmoke_%j.out
#SBATCH --error=compare_models/ab_dvsmoke_%j.err
#
# Force a hard divergence (absurd lr) with the scalar features and verify the new
# divergence-abort fires cleanly: ~diverge_patience consecutive skips -> RuntimeError
# -> non-zero exit, and NO CUDA device-side assert / index-OOB crash.

module load anaconda-py3/2023.09
source /gpfslocalsup/pub/anaconda-py3/2023.09/etc/profile.d/conda.sh
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
PROJ=$PWD

RUNDIR="$PROJ/compare_models/_ab_dvsmoke/run"; rm -rf "$RUNDIR"
echo "### diverge smoke: scalar features, lr=50 (forced blowup), diverge_patience=20 ###"
python run.py model=lloca local=none \
  data.source=recipes "data.processes_file=${PROJ}/recipes/scan_bigrun.yaml" \
  data.require_cache=true data.preprocess_per_dataset=true \
  data.train_subsample=2000 data.eval_subsample=500 data.seed=42 seed=42 \
  data.use_PIDs=false data.spin_onehot=true data.color_onehot=true \
  data.prop_is_massless=true data.standardize_props=true \
  data.mass_from_momenta=true data.coupling_scalars=true model.use_diagrams=false \
  training.batchsize=1024 evaluation.batchsize=4096 \
  training.loss_aggregation=geometric_mean training.regularization=L2 \
  training.regularization_lambda=1e-9 training.scheduler=CosineAnnealingLR \
  training.lr=50.0 training.iterations=1500 training.validate_frac=0.34 \
  training.diverge_patience=20 \
  training.get_ID=false training.dtype=float32 use_mlflow=false plot=false \
  exp_name="ab_dvsmoke" run_dir="$RUNDIR"
rc=$?
echo ">>> python exit code = $rc  (expect non-zero from clean RuntimeError, NOT a CUDA assert)"
