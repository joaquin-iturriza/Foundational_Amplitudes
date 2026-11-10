#!/bin/bash
#SBATCH --job-name=het_fix
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --output=analysis/divergences/het_fix_%j.out
#SBATCH --error=analysis/divergences/het_fix_%j.out
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd $PROJECT_DIR
REC=$PROJECT_DIR/recipes/pretrain22_heldout_uug.yaml
python run.py exp_name=heterosc_foundation run_name=het_fixtest \
  data.source=recipes data.processes_file="$REC" \
  model.use_diagrams=false model.particle_encoder_hidden=0 \
  training.loss=HETEROSC training.heterosc_beta=1.0 \
  training.batchsize=16384 training.iterations=2000 training.lr=0.004 \
  training.regularization_lambda=6.264093e-10 \
  training.cosanneal_warmup_frac=0.1079041 training.cosanneal_eta_min=6.17785e-9 \
  training.ema_decay=0.9330307 plot=true save=true
echo "DONE het_fixtest"
