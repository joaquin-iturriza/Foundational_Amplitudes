#!/bin/bash
#SBATCH --job-name=collision_ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=runs/_logs/collision_ab_%j.out
#SBATCH --error=runs/_logs/collision_ab_%j.err
#SBATCH --gres=gpu:1
#
# Encoding A/B on the flavour-collision recipe (recipes/collision_ab.yaml): one run with the
# generation one-hot on, one with it off, everything else the catalog_v2 sweep's fixed setup at
# a short horizon. Not an HP search: the only difference between the two jobs is the flag.
# GEN=false keeps the generation as a scalar column, which still separates d from s: the
# informative baseline is GEN=none (no column). At this scale both learn everything anyway;
# the collision only shows on the full recipe (scripts/job_catalog_short_ab.sh).
#   GEN=true  site submit <site> FA --env GEN=true  scripts/job_collision_ab.sh
#   GEN=false site submit <site> FA --env GEN=false scripts/job_collision_ab.sh
#   add LEVERS=off to either to drop the per-process physics levers (see below).
set -euo pipefail
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
GEN="${GEN:-true}"        # true: one-hot | false: scalar column (still separates d from s) | none: no column
if [ "$GEN" = "none" ]; then GFEAT=false; GHOT=false; else GFEAT=true; GHOT="$GEN"; fi
# LEVERS=off drops the per-process physics levers (off-shellness, internal masses, couplings),
# which are built from each process's propagator list and by themselves separate dd->dd from
# ds->ds at small scale; off, the twins' inputs are identical unless the generation column is on.
LEVERS="${LEVERS:-on}"
if [ "$LEVERS" = "off" ]; then LEV="false"; else LEV="true"; fi
python run.py \
  exp_name="collision_ab_gen_${GEN}_levers_${LEVERS}" \
  data.source=recipes data.require_cache=true data.seed=42 \
  data.processes_file="$PROJECT_DIR"/recipes/collision_ab.yaml \
  data.train_subsample=null data.eval_subsample=2000 data.preprocess_per_dataset=true \
  data.use_PIDs=false data.spin_onehot=true data.color_onehot=true data.prop_is_massless=true \
  data.standardize_props=true data.generation_onehot="${GHOT}" data.generation_feature="${GFEAT}" \
  data.mass_from_momenta=true data.coupling_scalars="$LEV" data.internal_mass_scalars="$LEV" \
  data.offshell_per_event="$LEV" 'data.internal_mass_pdgs=[23,6,25]' \
  model=lloca model.use_diagrams=false model.particle_encoder_hidden=0 \
  model.net.num_heads=8 model.net.num_blocks=8 seed=42 \
  training.iterations=3000 training.batchsize=4096 evaluation.batchsize=4096 \
  training.lr=4.1e-3 training.regularization_lambda=1e-8 training.cosanneal_warmup_frac=0.1 \
  training.loss_aggregation=geometric_mean training.regularization=L2 \
  training.scheduler=CosineAnnealingLR training.get_ID=false training.save_intermediate=false \
  training.validate_frac=0.02 evaluation.train_subsample=2000 training.dtype=float32 plot=true use_mlflow=false
