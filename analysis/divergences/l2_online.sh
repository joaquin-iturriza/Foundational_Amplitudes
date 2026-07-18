#!/bin/bash
#SBATCH --job-name=l2_online
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=analysis/divergences/l2_online_%x_%j.out
#SBATCH --error=analysis/divergences/l2_online_%x_%j.out
# Usage: sbatch --job-name=l2_<arm> l2_online.sh --arm <arm> [--rounds N --iters N ...]
# Runs the whole online L2 loop for ONE arm inside ONE GPU allocation (each GPU stage is its
# own fresh run.py/py subprocess to avoid the muP cross-instance global leak).
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-l2-gen
python analysis/divergences/l2_online.py "$@"
echo "DONE l2_online $*"
