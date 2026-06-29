#!/bin/bash
#SBATCH --job-name=scan_ab_repro
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --array=0-5
#SBATCH --output=compare_models/scan_ab_repro_%A_%a.out
#SBATCH --error=compare_models/scan_ab_repro_%A_%a.err
#
# Reproducibility check for the A/B diagram result: the ~0.60 val_loss_no_reg was
# an ISOLATED lr=1e-3 minimum (neighbours sat at baseline ~0.68), so re-run
# diagram AND off at lr=1e-3 across 3 fresh seeds. If diagram is consistently
# ~0.60 and off ~0.68, the diagram win is real; if diagram scatters, it was luck.

module load anaconda-py3/2023.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
PROJ=$PWD

ARMS=(diagram off)
SEEDS=(1 2 3)
ai=$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))
si=$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))
ARM=${ARMS[$ai]}; SEED=${SEEDS[$si]}

case "$ARM" in
  off)     FEAT="data.mass_from_momenta=false data.coupling_scalars=false model.use_diagrams=false" ;;
  diagram) FEAT="data.mass_from_momenta=true  data.coupling_scalars=false model.use_diagrams=true model.d_diag=32" ;;
esac

RUNDIR="$PROJ/compare_models/_scan_ab_repro/${ARM}_seed${SEED}"
rm -rf "$RUNDIR"
echo "### REPRO arm=$ARM lr=1e-3 seed=$SEED ###"
python run.py \
  model=lloca local=none \
  data.source=recipes "data.processes_file=${PROJ}/recipes/scan_bigrun.yaml" \
  data.require_cache=true data.preprocess_per_dataset=true \
  data.train_subsample=2000 data.eval_subsample=1000 \
  data.seed=42 seed=$SEED \
  data.use_PIDs=false data.spin_onehot=true data.color_onehot=true \
  data.prop_is_massless=true data.standardize_props=true \
  $FEAT \
  training.batchsize=1024 evaluation.batchsize=4096 \
  training.loss_aggregation=geometric_mean \
  training.regularization=L2 training.regularization_lambda=1e-8 \
  training.scheduler=CosineAnnealingLR training.lr=1e-3 \
  training.iterations=10000 training.validate_frac=0.1 \
  training.get_ID=false training.dtype=float32 use_mlflow=false plot=true \
  exp_name="scan_ab_repro_${ARM}_seed${SEED}" run_dir="$RUNDIR" \
  && echo ">>> $ARM seed=$SEED exited 0" || echo ">>> $ARM seed=$SEED FAILED"
