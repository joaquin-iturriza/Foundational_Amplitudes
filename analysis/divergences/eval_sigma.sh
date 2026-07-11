#!/bin/bash
#SBATCH --job-name=ev_sig
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=analysis/divergences/ev_sig_%j.out
#SBATCH --error=analysis/divergences/ev_sig_%j.out
set -e
module purge
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-heterosc
python analysis/divergences/extract_sigma.py
echo "DONE eval_sigma"
