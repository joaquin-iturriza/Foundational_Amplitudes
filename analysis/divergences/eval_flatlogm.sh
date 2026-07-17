#!/bin/bash
#SBATCH --job-name=ev_flatlogm
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=analysis/divergences/ev_flatlogm_%j.out
#SBATCH --error=analysis/divergences/ev_flatlogm_%j.out

set -e
module purge
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

# eval script lives on the heterosc branch (worktree only); run configs + ckpts are in MAIN.
# Three-way: raw RAMBO baseline vs flatlogm (RESAMPLED fixed pool -> coverage ceiling, fails)
# vs genflat (GENERATED fresh flat-log|M|^2 events -> breaks the ceiling). Each de-standardized
# with its OWN per-pool stats back to true log|M|^2 (comparable), binned by sqrt(s).
python worktrees/wt-heterosc/analysis/divergences/eval_eeuu_resonance.py \
  --runs_root runs/eeuu_flatlogm --run_prefix ft_ --tags raw,flatlogm,genflat

echo "DONE eval_flatlogm"
