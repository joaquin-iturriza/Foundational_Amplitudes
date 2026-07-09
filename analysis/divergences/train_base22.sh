#!/bin/bash
#SBATCH --job-name=base22_uug
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --output=analysis/divergences/base22_%j.out
#SBATCH --error=analysis/divergences/base22_%j.out

set -e
module purge
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

REC=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/recipes/pretrain22_heldout_uug.yaml

# 22-process leave-uug-out foundation base. Config = canonical recipe-pretrain
# defaults + the ADOPTED architecture candidate (docs/results.tex sec:arch):
# no diagram encoder + linear (no-MLP) particle embed -- the diagram encoder
# actively hurts (-35%) at scale and doubles step time. Only best-defaults changed:
#   iterations 10000, lr 2e-3, warmup 0.15, eta_min 1e-8, reg 1e-8.
python run.py \
  exp_name=pretrain22_heldout_uug run_name=base \
  data.source=recipes data.processes_file="$REC" \
  model.use_diagrams=false model.particle_encoder_hidden=0 \
  training.iterations=10000 training.lr=0.002 \
  training.cosanneal_warmup_frac=0.15 training.cosanneal_eta_min=1e-8 \
  training.regularization_lambda=1e-8 \
  plot=true save=true

echo "DONE base22"
