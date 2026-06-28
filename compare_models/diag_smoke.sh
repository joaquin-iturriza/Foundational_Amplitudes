#!/bin/bash
#SBATCH --job-name=diag_smoke
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=compare_models/diag_smoke_%j.out
#SBATCH --error=compare_models/diag_smoke_%j.err
#
# End-to-end smoke of Feynman-diagram conditioning on the μP LLoCa model: the full
# train->validate->save->plot path on the tiny 8-process short recipe, ~40 steps,
# with the real xformers attention kernel (CUDA-only) + the diagram graph encoder
# forward/backward. Runs the diagram-ON path and a no-diagram twin (off-path must
# still work). Confirms: registry load, encoder build, μP finalize, training step,
# checkpoint save, plotting — before any real diagram-conditioned training.

module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
PROJ=$PWD

COMMON=(
  model=lloca
  local=none
  data.source=recipes
  "data.processes_file=${PROJ}/recipes/pretrain8_short.yaml"
  "data.data_path=${PROJ}/data/"
  data.preprocess_per_dataset=true
  data.seed=42
  seed=42
  # smart particle encoding (use_PIDs stays false -> diagrams allowed)
  data.spin_onehot=true
  data.color_onehot=true
  data.prop_is_massless=true
  data.standardize_props=true
  training.batchsize=1024
  evaluation.batchsize=4096
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
  local RUNDIR="$PROJ/compare_models/_diag_smoke_${NAME}"
  rm -rf "$RUNDIR"
  echo "############################## SMOKE $NAME ##############################"
  python run.py "${COMMON[@]}" "$@" exp_name="diag_smoke_${NAME}" run_dir="$RUNDIR" \
    && echo ">>> $NAME: python exited 0" || echo ">>> $NAME: python FAILED"
}

# diagram conditioning ON (the new path)
run_one diag_on  model.use_diagrams=true  model.d_diag=32
# no-diagram twin (off-path must be unaffected)
run_one diag_off model.use_diagrams=false

echo "===== DIAG SMOKE DONE ====="
for N in diag_on diag_off; do
  D="$PROJ/compare_models/_diag_smoke_${N}"
  echo "[$N] models: $(ls $D/models 2>/dev/null | tr '\n' ' ')"
  echo "[$N] plots:  $(ls $D/plots  2>/dev/null | head -3 | tr '\n' ' ')"
done
