#!/bin/bash
#SBATCH --job-name=div_extract
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=analysis/divergences/extract_%j.out
#SBATCH --error=analysis/divergences/extract_%j.out

set -e
module purge
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational

cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

echo "=== eeuu NLO virt: extract ==="
python analysis/divergences/extract_preds.py \
  --run_dir runs/finetune_scaling_virt_002_eeuunlovirte4_t40000/trial_0154 \
  --out analysis/divergences/preds_eeuu_nlo.npz

echo "=== eett NLO virt: extract ==="
python analysis/divergences/extract_preds.py \
  --run_dir runs/finetune_scaling_virt_002_eettbarnlovirte4_t40000/trial_0154 \
  --out analysis/divergences/preds_eett_nlo.npz

echo "=== plots ==="
python analysis/divergences/make_plots.py \
  --npz analysis/divergences/preds_eeuu_nlo.npz \
  --label 'NLO $e^+e^-\to u\bar u$ (virtual)' \
  --out_base analysis/divergences/figs/phase_space_eeuu_nlo --split all
python analysis/divergences/make_plots.py \
  --npz analysis/divergences/preds_eett_nlo.npz \
  --label 'NLO $e^+e^-\to t\bar t$ (virtual)' \
  --out_base analysis/divergences/figs/phase_space_eett_nlo --split all

echo "DONE"
