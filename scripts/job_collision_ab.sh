#!/bin/bash
#SBATCH --job-name=collision_ab
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --account=lpnhe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=runs/_logs/collision_ab_%j.out
#SBATCH --error=runs/_logs/collision_ab_%j.err
#
# Encoding A/B on the flavour-collision recipe (recipes/collision_ab.yaml): one run with the
# generation one-hot on, one with it off, everything else the catalog_v2 sweep's fixed setup at
# a short horizon. Not an HP search: the only difference between the two jobs is the flag.
#   GEN=true  scripts/remote.sh sbatch --export=ALL,GEN=true  scripts/job_collision_ab.sh
#   GEN=false scripts/remote.sh sbatch --export=ALL,GEN=false scripts/job_collision_ab.sh
set -euo pipefail
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes
GEN="${GEN:-true}"
python run.py \
  exp_name="collision_ab_gen_${GEN}" \
  data.source=recipes data.require_cache=true data.seed=42 \
  data.processes_file=/sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/recipes/collision_ab.yaml \
  data.train_subsample=null data.eval_subsample=2000 data.preprocess_per_dataset=true \
  data.use_PIDs=false data.spin_onehot=true data.color_onehot=true data.prop_is_massless=true \
  data.standardize_props=true data.generation_onehot="${GEN}" \
  data.mass_from_momenta=true data.coupling_scalars=true data.internal_mass_scalars=true \
  data.offshell_per_event=true 'data.internal_mass_pdgs=[23,6,25]' \
  model=lloca model.use_diagrams=false model.particle_encoder_hidden=0 \
  model.net.num_heads=8 model.net.num_blocks=8 seed=42 \
  training.iterations=3000 training.batchsize=4096 evaluation.batchsize=4096 \
  training.lr=4.1e-3 training.regularization_lambda=1e-8 training.cosanneal_warmup_frac=0.1 \
  training.loss_aggregation=geometric_mean training.regularization=L2 \
  training.scheduler=CosineAnnealingLR training.get_ID=false training.save_intermediate=false \
  training.validate_frac=0.02 training.dtype=float32 plot=true use_mlflow=false
