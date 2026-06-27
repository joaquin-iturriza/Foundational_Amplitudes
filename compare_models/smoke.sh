#!/bin/bash
#SBATCH --job-name=amp_smoke
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=compare_models/smoke_%j.out
#SBATCH --error=compare_models/smoke_%j.err
#
# End-to-end smoke of the full train->validate->save->plot path for both geometric
# models, ~40 steps on a small subsample, before launching the 40-trial sweeps.

module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
PROJ=$PWD

COMMON=(
  local=none
  data.source=recipes
  "data.processes_file=${PROJ}/recipes/pretrain25.yaml"
  "data.data_path=${PROJ}/data/"
  data.preprocess_per_dataset=true
  data.require_cache=true
  data.train_subsample=8000
  data.eval_subsample=4000
  data.seed=42
  seed=42
  training.batchsize=1024
  evaluation.batchsize=8192
  training.loss_aggregation=geometric_mean
  training.regularization=L2
  training.regularization_lambda=1e-8
  training.scheduler=CosineAnnealingLR
  training.lr=2e-3
  training.iterations=40
  training.validate_frac=0.25
  training.save_intermediate=true
  training.get_ID=false
  training.dtype=float32
  use_mlflow=false
  plot=true
)

run_one () {
  local NAME=$1; shift
  local RUNDIR="$PROJ/compare_models/_smoke_${NAME}"
  rm -rf "$RUNDIR"
  echo "############################## SMOKE $NAME ##############################"
  python run.py "${COMMON[@]}" "$@" exp_name="smoke_${NAME}" run_dir="$RUNDIR" \
    && echo ">>> $NAME: python exited 0" || echo ">>> $NAME: python FAILED"
}

run_one lgatr model=lgatr_mup  model.net.num_blocks=8 \
                model.net.hidden_mv_channels=22 model.net.hidden_s_channels=22
run_one slim  model=lgatr_slim model.net.num_blocks=8 \
                model.net.hidden_v_channels=52 model.net.hidden_s_channels=104

echo "===== SMOKE DONE ====="
for N in lgatr slim; do
  D="$PROJ/compare_models/_smoke_${N}"
  echo "[$N] models: $(ls $D/models 2>/dev/null | tr '\n' ' ')"
  echo "[$N] plots:  $(ls $D/plots  2>/dev/null | head -3 | tr '\n' ' ')"
done
