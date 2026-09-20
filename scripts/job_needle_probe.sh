#!/bin/bash
#SBATCH --job-name=needle_probe
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --account=lpnhe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=runs/_logs/needle_probe_%j.out
#SBATCH --error=runs/_logs/needle_probe_%j.err
#
# Needle probe (recipes/needle_probe.yaml): the s-channel 2->2 family alone, short horizon,
# eval on the full 10k validation pools so the Z-pole region has ~260 events per process;
# final-step per-event predictions are saved (preds/) for binning the residual in sqrt(s).
#   scripts/remote.sh sbatch --export=ALL,GEN=true,EXP=needle_probe scripts/job_needle_probe.sh
# AGG=mean|geometric_mean and TAU=<float> (tau-floored geometric mean, training loss only)
# switch the training aggregation; the validation metric stays the geometric mean.
# TRAIN_SUB= caps the train events per process. BS= and STEPS= set the batch size and horizon (solo reference runs: BS=34, the joint run's bs/P).
set -euo pipefail
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes
GEN="${GEN:-true}"        # true: one-hot | false: scalar column (still separates d from s) | none: no column
if [ "$GEN" = "none" ]; then GFEAT=false; GHOT=false; else GFEAT=true; GHOT="$GEN"; fi
# LEVERS=off drops the per-process physics levers (off-shellness, internal masses, couplings),
# which are built from each process's propagator list and by themselves separate dd->dd from
# ds->ds at small scale; off, the twins' inputs are identical unless the generation column is on.
LEVERS="${LEVERS:-on}"
if [ "$LEVERS" = "off" ]; then LEV="false"; else LEV="true"; fi
python run.py \
  exp_name="${EXP:-needle_probe}" \
  data.source=recipes data.require_cache=true data.seed=42 \
  data.processes_file=/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/recipes/${RECIPE:-needle_probe.yaml} \
  data.train_subsample="${TRAIN_SUB:-null}" data.eval_subsample=10000 data.preprocess_per_dataset=true \
  data.use_PIDs=false data.spin_onehot=true data.color_onehot=true data.prop_is_massless=true \
  data.standardize_props=true data.generation_onehot="${GHOT}" data.generation_feature="${GFEAT}" \
  data.mass_from_momenta=true data.coupling_scalars="$LEV" data.internal_mass_scalars="$LEV" \
  data.offshell_per_event="$LEV" 'data.internal_mass_pdgs=[23,6,25]' \
  model=lloca model.use_diagrams=false model.particle_encoder_hidden=0 \
  model.net.num_heads=8 model.net.num_blocks=8 seed=42 \
  training.iterations="${STEPS:-3000}" training.batchsize="${BS:-4096}" evaluation.batchsize=4096 \
  training.lr=4.1e-3 training.regularization_lambda=1e-8 training.cosanneal_warmup_frac=0.1 \
  training.loss_aggregation="${AGG:-geometric_mean}" training.loss_aggregation_tau="${TAU:-0.0}" \
  training.regularization=L2 \
  training.scheduler=CosineAnnealingLR training.get_ID=false training.save_intermediate=false \
  training.validate_frac=0.02 evaluation.train_subsample=2000 training.dtype=float32 plot=true use_mlflow=false
