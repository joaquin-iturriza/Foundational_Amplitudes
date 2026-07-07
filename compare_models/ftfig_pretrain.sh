#!/bin/bash
#SBATCH --job-name=ftfig_pre
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=compare_models/ftfig_pre_%x_%j.out
#SBATCH --error=compare_models/ftfig_pre_%x_%j.err
# Pretrained models for the fine-tune compute-scan figure (leak-free recipes:
# 416 = bigrun minus ee_ttbar_nlo*/ee_cc_nlo* [full-NLO twins of the two virt
# fine-tune targets], lo352 = LO only). $RECIPE + $ARM via --export; the feature
# set is selected from $ARM below (an --export value cannot carry commas, so
# flag lists live here). HPs = bigrun best trial; ITERS defaults to the bigrun
# 8601 (override for 1h ladder rungs). Ladder rungs (1h, 416 sets):
#   raw416 -> rung2 (+onehots/standardize) -> rung3 (+mass/coupling) -> best416
# (rung4 == best416 flags at ITERS=3400; raw rung == raw416 at ITERS=3400).
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
export AMP_FROZEN_DIR=$SCRATCH/datasets_scanbig_cut
export AMP_TRAIN_CACHE_DIR=$SCRATCH/amp_cache_scanbig_cut
export AMP_FIDUCIAL_CUTS=on
ITERS=${ITERS:-8601}

# Cumulative feature ladder; each case lists the FULL flag set (no duplicate
# Hydra keys). ONEHOT = smart encoding; PHYS = derived per-particle scalars;
# FULL = internal-mass slot + per-event off-shellness (the adopted candidate).
OH=true; PH=true; FU=true
case "$ARM" in
  raw*)   OH=false; PH=false; FU=false ;;
  rung2*) PH=false; FU=false ;;
  rung3*) FU=false ;;
  best*|lo*) ;;
  *) echo "unknown ARM=$ARM"; exit 1 ;;
esac
ARM_OVR="data.spin_onehot=$OH data.color_onehot=$OH data.prop_is_massless=$OH
  data.standardize_props=$OH data.mass_from_momenta=$PH data.coupling_scalars=$PH
  data.internal_mass_scalars=$FU data.offshell_per_event=$FU
  model.use_diagrams=false model.particle_encoder_hidden=0"
[ "$FU" = true ] && ARM_OVR="$ARM_OVR data.internal_mass_pdgs=[23,6,25]"
python run.py model=lloca local=none \
  model.net.num_heads=8 model.net.num_blocks=8 \
  data.source=recipes data.processes_file=$PWD/recipes/${RECIPE}.yaml \
  data.require_cache=true data.preprocess_per_dataset=true \
  data.train_subsample=null data.eval_subsample=2000 data.seed=42 seed=42 \
  data.use_PIDs=false \
  training.batchsize=16384 evaluation.batchsize=16384 \
  training.use_balanced_sampler=false \
  training.loss_aggregation=geometric_mean training.regularization=L2 \
  training.scheduler=CosineAnnealingLR training.dtype=float32 \
  training.get_ID=false use_mlflow=false \
  training.lr=4.079396e-03 training.regularization_lambda=7.001527e-07 \
  training.cosanneal_warmup_frac=1.494990e-01 training.cosanneal_eta_min=5.142869e-10 \
  ema=true training.ema_decay=9.881993e-01 training.iterations=$ITERS \
  evaluate=false plot=true \
  training.result_path=$PWD/compare_models/_ftfig_pre_${ARM}/result.json \
  $ARM_OVR \
  exp_name=ftfig_pre_${ARM} run_dir=$PWD/compare_models/_ftfig_pre_${ARM}
echo "===== FTFIG PRETRAIN $ARM DONE ====="
