#!/bin/bash
#SBATCH --job-name=mass_tau
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --array=0-1
#SBATCH --output=compare_models/mass_tau_%A_%a.out
#SBATCH --error=compare_models/mass_tau_%A_%a.err
#
# Mass honest test: off vs data.mass_from_momenta on near-threshold wide-m_t
# ee_ttbar (private cache). Production per-dataset standardization (the mass effect
# is a per-event threshold SHAPE, so it survives). Same HP both arms (off-best).

module load anaconda-py3/2023.09
source /gpfslocalsup/pub/anaconda-py3/2023.09/etc/profile.d/conda.sh
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
PROJ=$PWD
export AMP_TRAIN_CACHE_DIR=$SCRATCH/amp_cache_masstau
export AMP_FROZEN_DIR=$SCRATCH/datasets_masstau

if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then N=off;  M=false; else N=mass; M=true; fi
RUNDIR="$PROJ/compare_models/_mass_tau/$N"; rm -rf "$RUNDIR"
echo "### mass test $N (mass_from_momenta=$M) ###"
python run.py model=lloca local=none \
  data.source=recipes "data.processes_file=${PROJ}/recipes/mass_tautaua_test.yaml" \
  data.require_cache=true data.preprocess_per_dataset=true \
  data.eval_subsample=1000 data.seed=42 seed=42 \
  data.use_PIDs=false data.spin_onehot=true data.color_onehot=true \
  data.prop_is_massless=true data.standardize_props=true \
  data.mass_from_momenta=$M data.coupling_scalars=false model.use_diagrams=false \
  training.batchsize=1024 evaluation.batchsize=4096 \
  training.loss_aggregation=geometric_mean training.regularization=L2 \
  training.scheduler=CosineAnnealingLR \
  training.lr=0.0037850206125016168 training.regularization_lambda=1.4383791066176087e-10 \
  training.cosanneal_warmup_frac=0.037281612306833266 \
  training.cosanneal_eta_min=5.00942594074264e-10 training.ema_decay=0.9687763301244937 \
  training.iterations=4000 training.validate_frac=0.02 \
  training.get_ID=false training.dtype=float32 use_mlflow=false plot=true \
  exp_name="mass_test_$N" run_dir="$RUNDIR" \
  && echo ">>> $N done; val + plots in $RUNDIR/plots_0" || echo ">>> $N FAILED"
