#!/bin/bash
#SBATCH --job-name=smoke_l2sig
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=analysis/divergences/smoke_l2sig_%j.out
#SBATCH --error=analysis/divergences/smoke_l2sig_%j.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-l2-uugg
export PYTHONDONTWRITEBYTECODE=1
python analysis/divergences/l2_online_uugg.py \
  --arm sigma --tag smoke --total_steps 12 --rounds 3 --n_total 1500 \
  --oversample 6 --gamma 1.0 --y_lo 1e-6 --mix_ir 0.5 --seed 0
echo "DONE smoke sigma"
