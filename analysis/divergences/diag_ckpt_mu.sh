#!/bin/bash
#SBATCH --job-name=diag_mu
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=analysis/divergences/diag_mu_%j.out
#SBATCH --error=analysis/divergences/diag_mu_%j.out
set -e
module purge
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-heterosc
python analysis/divergences/diag_ckpt_mu.py
echo "DONE diag_ckpt_mu"
