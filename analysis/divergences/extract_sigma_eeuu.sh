#!/bin/bash
#SBATCH --job-name=ext_sig_eeuu
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=analysis/divergences/ext_sig_eeuu_%j.out
#SBATCH --error=analysis/divergences/ext_sig_eeuu_%j.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
python worktrees/wt-heterosc/analysis/divergences/extract_sigma_eeuu.py --run_dir worktrees/wt-heterosc/runs/eeuu_sigfit/mix025_sigma
echo "DONE extract_sigma_eeuu"
