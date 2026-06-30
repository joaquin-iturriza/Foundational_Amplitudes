#!/bin/bash
#SBATCH --job-name=ab_bestplot
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --array=0-2
#SBATCH --output=compare_models/ab_bestplot_%A_%a.out
#SBATCH --error=compare_models/ab_bestplot_%A_%a.err
#
# Re-run the BEST HP config of each A/B arm with plot=true so the run dirs finally
# have plots (the sweeps wrongly ran plot=false). Same data/preproc as the sweep.

module load anaconda-py3/2023.09
source /gpfslocalsup/pub/anaconda-py3/2023.09/etc/profile.d/conda.sh
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
PROJ=$PWD

# Read from scan_bigrun's PRIVATE cache (decoupled from the shared cache that the
# concurrent pretrain25_zs_phys sweep clobbers — see datagen.train_cache_dir /
# frozen_dir overrides). Must match the dirs the private prebuild wrote into.
export AMP_TRAIN_CACHE_DIR=$SCRATCH/amp_data_cache_scanbig
export AMP_FROZEN_DIR=$SCRATCH/datasets_scanbig

case "$SLURM_ARRAY_TASK_ID" in
  0) N=off;     FEAT="data.mass_from_momenta=false data.coupling_scalars=false model.use_diagrams=false"
     HP="training.lr=0.0037850206125016168 training.regularization_lambda=1.4383791066176087e-10 training.cosanneal_warmup_frac=0.037281612306833266 training.cosanneal_eta_min=5.00942594074264e-10 training.ema_decay=0.9687763301244937" ;;
  1) N=scalar;  FEAT="data.mass_from_momenta=true data.coupling_scalars=true model.use_diagrams=false"
     HP="training.lr=0.006414088841184999 training.regularization_lambda=7.706949012930404e-08 training.cosanneal_warmup_frac=0.032987560518085955 training.cosanneal_eta_min=8.402842694321212e-11 training.ema_decay=0.910914858643245" ;;
  2) N=diagram; FEAT="data.mass_from_momenta=true data.coupling_scalars=false model.use_diagrams=true"
     HP="training.lr=0.0063282981750705765 training.regularization_lambda=1.6674095155451647e-08 training.cosanneal_warmup_frac=0.16586464829742908 training.cosanneal_eta_min=1.882093155023226e-09 training.ema_decay=0.9798943646321073" ;;
esac
RUNDIR="$PROJ/compare_models/_ab_bestplots/$N"; rm -rf "$RUNDIR"
echo "### bestplot $N ###"
python run.py model=lloca local=none \
  data.source=recipes "data.processes_file=${PROJ}/recipes/scan_bigrun.yaml" \
  data.require_cache=true data.preprocess_per_dataset=true \
  data.train_subsample=2000 data.eval_subsample=500 data.seed=42 seed=42 \
  data.use_PIDs=false data.spin_onehot=true data.color_onehot=true \
  data.prop_is_massless=true data.standardize_props=true \
  $FEAT \
  training.batchsize=1024 evaluation.batchsize=4096 \
  training.loss_aggregation=geometric_mean training.regularization=L2 \
  training.scheduler=CosineAnnealingLR $HP \
  training.iterations=5000 training.validate_frac=0.34 \
  training.get_ID=false training.dtype=float32 use_mlflow=false plot=true \
  exp_name="ab_bestplot_$N" run_dir="$RUNDIR" \
  && echo ">>> $N done; plots in $RUNDIR/plots" || echo ">>> $N FAILED"
