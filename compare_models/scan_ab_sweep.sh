#!/bin/bash
#SBATCH --job-name=scan_ab
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --array=0-17
#SBATCH --output=compare_models/scan_ab_%A_%a.out
#SBATCH --error=compare_models/scan_ab_%A_%a.err
#
# 3-way coupling+mass A/B on the full 385-dataset scan set.
#   arms : off       (no coupling/mass info)
#          scalar    (mass_from_momenta + global coupling scalar)
#          diagram   (mass_from_momenta + per-vertex coupling via diagrams)
# Each arm gets its OWN short lr sweep (6 lrs); compare best-vs-best on the
# non-regularized val loss (val_loss_no_reg) saved in per_process_metrics.json.

module load anaconda-py3/2023.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
PROJ=$PWD

ARMS=(off scalar diagram)
LRS=(3e-4 6e-4 1e-3 2e-3 4e-3 8e-3)
ai=$(( SLURM_ARRAY_TASK_ID / ${#LRS[@]} ))
li=$(( SLURM_ARRAY_TASK_ID % ${#LRS[@]} ))
ARM=${ARMS[$ai]}; LR=${LRS[$li]}

case "$ARM" in
  off)      FEAT="data.mass_from_momenta=false data.coupling_scalars=false model.use_diagrams=false" ;;
  scalar)   FEAT="data.mass_from_momenta=true  data.coupling_scalars=true  model.use_diagrams=false" ;;
  diagram)  FEAT="data.mass_from_momenta=true  data.coupling_scalars=false model.use_diagrams=true model.d_diag=32" ;;
esac

RUNDIR="$PROJ/compare_models/_scan_ab/${ARM}_lr${LR}"
rm -rf "$RUNDIR"
echo "### A/B arm=$ARM lr=$LR -> $RUNDIR ###"
python run.py \
  model=lloca local=none \
  data.source=recipes "data.processes_file=${PROJ}/recipes/scan_bigrun.yaml" \
  data.require_cache=true data.preprocess_per_dataset=true \
  data.train_subsample=2000 data.eval_subsample=1000 \
  data.seed=42 seed=42 \
  data.use_PIDs=false data.spin_onehot=true data.color_onehot=true \
  data.prop_is_massless=true data.standardize_props=true \
  $FEAT \
  training.batchsize=1024 evaluation.batchsize=4096 \
  training.loss_aggregation=geometric_mean \
  training.regularization=L2 training.regularization_lambda=1e-8 \
  training.scheduler=CosineAnnealingLR training.lr=$LR \
  training.iterations=10000 training.validate_frac=0.1 \
  training.get_ID=false training.dtype=float32 use_mlflow=false plot=true \
  exp_name="scan_ab_${ARM}_lr${LR}" run_dir="$RUNDIR" \
  && echo ">>> $ARM lr=$LR exited 0" || echo ">>> $ARM lr=$LR FAILED"
