#!/bin/bash
#SBATCH --job-name=catalog_short_ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=runs/_logs/catalog_short_ab_%j.out
#SBATCH --error=runs/_logs/catalog_short_ab_%j.err
#SBATCH --gres=gpu:1
#
# Encoding A/B at the catalog's own scale: the full catalog_v2 train+scan recipe (478 pools),
# the wave-0 sweep setup, 1000 steps (the ds->ds one-loop anti-learning of the census showed
# by step ~900), generation one-hot (GEN=true) vs no generation column at all (GEN=none, the
# encoding the census ran with). GEN=false keeps the column as a scalar. Not an HP search.
# EXP= names the run; AGG=mean|geometric_mean and TAU=<float> switch the training aggregation
# AGG=excess needs REF=4=<L_ref>_5=<L_ref>_6=<L_ref> (solo loss per particle count; underscores,
# since sbatch --export splits on commas) and
# takes BETA= (training.excess_beta). HEADS= sets the width (num_heads, the muP axis; lr transfers).
# TRAIN_SUB= caps the train events per process (small-pool, data-limited runs). LR= sets the learning rate; the default 5.8e-3 is the horizon law's centre at 1000 steps
# (lr_c(t) = 1e-2 (t/3000)^0.5 below t* = 3000; the earlier arms ran at the 8601-step best, 9.7e-3).
# (the validation metric stays the geometric mean). SLQ= sets data.signedlog_quantile (the
# scale of the signed log for sign-changing pools; 0.01 default, 0.5 = the median).
# STEPS= sets the horizon (default 1000), VF= the validation fraction (default 0.05), EVBS= the evaluation batch size.
# LAM= and WU= set the regularization and warm-up fraction; EXTRA= appends further overrides verbatim.
# SEED= sets the init seed (repeats; data.seed stays 42).
# RECIPE= picks the recipe file under recipes/ (default the full catalog); BS= the training batch size.
# TPROP=true divides the massive s-channel propagators out of the target, TCH=true adds the t-channel
# factor, TCHMAX= its max final-state count (2 = the adopted 2->2-only rule), SIGN=true the sign head.
# (Separate variables because an EXTRA with spaces does not survive the --export of `site submit`.)
# ETA= sets training.cosanneal_eta_min, EMA=true|false the weight EMA and EMAD= its decay.
set -euo pipefail
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
GEN="${GEN:-true}"        # true: one-hot | false: scalar column | none: no generation column (the old encoding)
if [ "$GEN" = "none" ]; then GFEAT=false; GHOT=false; else GFEAT=true; GHOT="$GEN"; fi
python run.py \
  exp_name="${EXP:-catalog_short_ab_gen_${GEN}}" \
  data.source=recipes data.require_cache=true data.seed=42 \
  data.processes_file="$PROJECT_DIR"/recipes/${RECIPE:-catalog_v2_train_scan.yaml} \
  data.train_subsample="${TRAIN_SUB:-null}" data.eval_subsample=2000 data.preprocess_per_dataset=true \
  data.signedlog_quantile="${SLQ:-0.01}" \
  data.target_propagators="${TPROP:-false}" data.target_propagator_tchannel="${TCH:-false}" \
  data.target_propagator_tchannel_max_final="${TCHMAX:-99}" training.sign_head="${SIGN:-false}" \
  data.use_PIDs=false data.spin_onehot=true data.color_onehot=true data.prop_is_massless=true \
  data.standardize_props=true data.generation_onehot="${GHOT}" data.generation_feature="${GFEAT}" \
  data.mass_from_momenta=true data.coupling_scalars=true data.internal_mass_scalars=true \
  data.offshell_per_event=true "data.internal_mass_pdgs=[$(echo "${PDGS:-23_6_25}" | tr _ ,)]" \
  model=lloca model.use_diagrams=false model.particle_encoder_hidden=0 \
  model.net.num_heads="${HEADS:-8}" model.net.num_blocks=8 seed="${SEED:-42}" \
  training.iterations="${STEPS:-1000}" training.batchsize="${BS:-16384}" evaluation.batchsize="${EVBS:-16384}" \
  training.lr="${LR:-5.8e-3}" training.regularization_lambda="${LAM:-1e-8}" training.cosanneal_warmup_frac="${WU:-0.1}" \
  training.loss_aggregation="${AGG:-geometric_mean}" training.loss_aggregation_tau="${TAU:-0.0}" \
  training.excess_beta="${BETA:-0.0}" ${REF:+"training.excess_reference={$(echo "$REF" | sed 's/_/,/g; s/=/:/g')}"} \
  training.regularization=L2 \
  training.cosanneal_eta_min="${ETA:-0}" ema="${EMA:-false}" training.ema_decay="${EMAD:-0.99}" \
  training.scheduler=CosineAnnealingLR training.get_ID=false training.save_intermediate=false \
  training.validate_frac="${VF:-0.05}" evaluation.train_subsample=2000 training.dtype=float32 plot=true use_mlflow=false ${EXTRA:-}
