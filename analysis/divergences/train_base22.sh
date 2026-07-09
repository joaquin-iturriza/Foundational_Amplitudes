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

# 22-process leave-uug-out foundation base. HPs copied verbatim from the well-trained
# 25ds reference runs/pretrain25/trial_0009 (recipe path; the trial the ft25 finetunes
# were built on), with ONLY the decided best-practice architecture overrides applied:
# no diagram encoder + linear (no-MLP) embed (docs/results.tex sec:arch: diagram encoder
# actively hurts -35%). Physics levers stay default-off, as in pretrain25 (the offshell/
# coupling levers are a big-run addition, not 25ds).
# best-practice-fixed: batchsize 16384 (canonical operating point; NOT pretrain25's
#   stale 1024), lr = lr-law foundation centre (~2e-3), no diagrams + linear embed.
# copied from the 25ds ref (pretrain25/trial_0009), being within best-practice ranges:
#   reg_lambda, warmup_frac, eta_min, ema_decay.
python run.py \
  exp_name=pretrain22_heldout_uug run_name=base \
  data.source=recipes data.processes_file="$REC" \
  model.use_diagrams=false model.particle_encoder_hidden=0 \
  training.batchsize=16384 training.iterations=10000 training.lr=0.004 \
  training.regularization_lambda=6.264093e-10 \
  training.cosanneal_warmup_frac=0.1079041 training.cosanneal_eta_min=6.17785e-9 \
  training.ema_decay=0.9330307 \
  plot=true save=true

echo "DONE base22"
