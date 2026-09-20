#!/bin/bash
#SBATCH --job-name=catalog_short_ab
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --account=lpnhe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=runs/_logs/catalog_short_ab_%j.out
#SBATCH --error=runs/_logs/catalog_short_ab_%j.err
#
# Encoding A/B at the catalog's own scale: the full catalog_v2 train+scan recipe (478 pools),
# the wave-0 sweep setup, 1000 steps (the ds->ds one-loop anti-learning of the census showed
# by step ~900), generation one-hot (GEN=true) vs no generation column at all (GEN=none, the
# encoding the census ran with). GEN=false keeps the column as a scalar. Not an HP search.
# EXP= names the run; AGG=mean|geometric_mean and TAU=<float> switch the training aggregation
# AGG=excess needs REF=4=<L_ref>_5=<L_ref>_6=<L_ref> (solo loss per particle count; underscores,
# since sbatch --export splits on commas) and
# takes BETA= (training.excess_beta).
# (the validation metric stays the geometric mean). SLQ= sets data.signedlog_quantile (the
# scale of the signed log for sign-changing pools; 0.01 default, 0.5 = the median).
set -euo pipefail
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes
GEN="${GEN:-true}"        # true: one-hot | false: scalar column | none: no generation column (the old encoding)
if [ "$GEN" = "none" ]; then GFEAT=false; GHOT=false; else GFEAT=true; GHOT="$GEN"; fi
python run.py \
  exp_name="${EXP:-catalog_short_ab_gen_${GEN}}" \
  data.source=recipes data.require_cache=true data.seed=42 \
  data.processes_file=/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/recipes/catalog_v2_train_scan.yaml \
  data.train_subsample=null data.eval_subsample=2000 data.preprocess_per_dataset=true \
  data.signedlog_quantile="${SLQ:-0.01}" \
  data.use_PIDs=false data.spin_onehot=true data.color_onehot=true data.prop_is_massless=true \
  data.standardize_props=true data.generation_onehot="${GHOT}" data.generation_feature="${GFEAT}" \
  data.mass_from_momenta=true data.coupling_scalars=true data.internal_mass_scalars=true \
  data.offshell_per_event=true 'data.internal_mass_pdgs=[23,6,25]' \
  model=lloca model.use_diagrams=false model.particle_encoder_hidden=0 \
  model.net.num_heads=8 model.net.num_blocks=8 seed=42 \
  training.iterations=1000 training.batchsize=16384 evaluation.batchsize=16384 \
  training.lr=9.7e-3 training.regularization_lambda=1e-8 training.cosanneal_warmup_frac=0.1 \
  training.loss_aggregation="${AGG:-geometric_mean}" training.loss_aggregation_tau="${TAU:-0.0}" \
  training.excess_beta="${BETA:-0.0}" ${REF:+"training.excess_reference={$(echo "$REF" | sed 's/_/,/g; s/=/:/g')}"} \
  training.regularization=L2 \
  training.scheduler=CosineAnnealingLR training.get_ID=false training.save_intermediate=false \
  training.validate_frac=0.05 evaluation.train_subsample=2000 training.dtype=float32 plot=true use_mlflow=false
