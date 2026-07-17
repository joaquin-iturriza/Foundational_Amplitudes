#!/bin/bash
#SBATCH --job-name=ev_l1
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=analysis/divergences/ev_l1_%j.out
#SBATCH --error=analysis/divergences/ev_l1_%j.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
python worktrees/wt-heterosc/analysis/divergences/eval_eeuu_resonance.py \
  --runs_root runs/eeuu_flatlogm --run_prefix ft_ \
  --tags mix025,l1base,l1sigma,l1oracle \
  --out_prefix eeuu_reson_l1_ --summary eeuu_reson_l1_summary.json
echo "DONE eval_l1"
