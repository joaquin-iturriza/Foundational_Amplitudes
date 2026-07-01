#!/bin/bash
#SBATCH --job-name=coup_st
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --array=0-1
#SBATCH --output=compare_models/coup_st_%A_%a.out
#SBATCH --error=compare_models/coup_st_%A_%a.err
#
# Coupling capability test: off vs data.coupling_scalars on flat wide-alpha
# ee_uuggg (identical kinematics, alpha only differs; private cache). POOLED
# standardization (preprocess_per_dataset=false) so the alpha-offset is NOT
# removed -> off CANNOT resolve alpha (identical kinematics+amp_orders across the
# 8 datasets); the coupling feature can. Same HP both arms (off-best).

module load anaconda-py3/2023.09
source /gpfslocalsup/pub/anaconda-py3/2023.09/etc/profile.d/conda.sh
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
PROJ=$PWD
export AMP_TRAIN_CACHE_DIR=$SCRATCH/amp_cache_coupsteep
export AMP_FROZEN_DIR=$SCRATCH/datasets_coupsteep

if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then N=off; C=false; else N=coupling; C=true; fi
RUNDIR="$PROJ/compare_models/_coup_steep/$N"; rm -rf "$RUNDIR"
echo "### coupling test $N (coupling_scalars=$C) ###"
python run.py model=lloca local=none \
  data.source=recipes "data.processes_file=${PROJ}/recipes/coupling_steep_test.yaml" \
  data.require_cache=true data.preprocess_per_dataset=true \
  data.eval_subsample=1000 data.seed=42 seed=42 \
  data.use_PIDs=false data.spin_onehot=true data.color_onehot=true \
  data.prop_is_massless=true data.standardize_props=true \
  data.mass_from_momenta=false data.coupling_scalars=$C model.use_diagrams=false \
  training.batchsize=1024 evaluation.batchsize=4096 \
  training.loss_aggregation=geometric_mean training.regularization=L2 \
  training.scheduler=CosineAnnealingLR \
  training.lr=0.0037850206125016168 training.regularization_lambda=1.4383791066176087e-10 \
  training.cosanneal_warmup_frac=0.037281612306833266 \
  training.cosanneal_eta_min=5.00942594074264e-10 training.ema_decay=0.9687763301244937 \
  training.iterations=4000 training.validate_frac=0.02 \
  training.get_ID=false training.dtype=float32 use_mlflow=false plot=true \
  exp_name="coup_test_$N" run_dir="$RUNDIR" \
  && echo ">>> $N done; val + plots in $RUNDIR/plots_0" || echo ">>> $N FAILED"
