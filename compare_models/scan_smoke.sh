#!/bin/bash
#SBATCH --job-name=scan_smoke
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=compare_models/scan_smoke_%j.out
#SBATCH --error=compare_models/scan_smoke_%j.err
#
# End-to-end smoke of the coupling+mass features on the μP LLoCa model: full
# recipe path (inline scan generation) -> train/validate/save/plot with the real
# xformers attention + diagram graph encoder. Confirms, on GPU, that the new
# inputs flow through the actual net:
#   - data.mass_from_momenta=true : per-particle on-shell mass replaces the table
#   - data.coupling_scalars=true  : global per-event log(alpha_QED/QCD) fallback
#   - model.use_diagrams=true     : per-vertex coupling-factor columns
#   - a coupling SCAN (ee_uug at two alpha_s) generated inline.
# Runs all features ON, plus an all-OFF twin (old path must be unaffected).

module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
PROJ=$PWD

COMMON=(
  model=lloca
  local=none
  data.source=recipes
  "data.processes_file=${PROJ}/recipes/scan_smoke.yaml"
  data.require_cache=false
  data.preprocess_per_dataset=true
  data.seed=42
  seed=42
  data.use_PIDs=false
  data.spin_onehot=true
  data.color_onehot=true
  data.prop_is_massless=true
  data.standardize_props=true
  training.batchsize=512
  evaluation.batchsize=2048
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
  local RUNDIR="$PROJ/compare_models/_scan_smoke_${NAME}"
  rm -rf "$RUNDIR"
  echo "############################## SMOKE $NAME ##############################"
  python run.py "${COMMON[@]}" "$@" exp_name="scan_smoke_${NAME}" run_dir="$RUNDIR" \
    && echo ">>> $NAME: python exited 0" || echo ">>> $NAME: python FAILED"
}

# all coupling+mass features ON (the new path)
run_one feat_on \
  data.mass_from_momenta=true \
  data.coupling_scalars=true \
  model.use_diagrams=true model.d_diag=32
# all OFF twin (old path must be unaffected; couplings ignored)
run_one feat_off \
  data.mass_from_momenta=false \
  data.coupling_scalars=false \
  model.use_diagrams=false

echo "===== SCAN SMOKE DONE ====="
for N in feat_on feat_off; do
  D="$PROJ/compare_models/_scan_smoke_${N}"
  echo "[$N] models: $(ls $D/models 2>/dev/null | tr '\n' ' ')"
  echo "[$N] plots:  $(ls $D/plots  2>/dev/null | head -3 | tr '\n' ' ')"
done
