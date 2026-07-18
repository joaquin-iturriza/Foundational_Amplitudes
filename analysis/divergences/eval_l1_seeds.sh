#!/bin/bash
#SBATCH --job-name=ev_l1sd
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=analysis/divergences/ev_l1sd_%j.out
#SBATCH --error=analysis/divergences/ev_l1sd_%j.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
python worktrees/wt-heterosc/analysis/divergences/eval_eeuu_resonance.py \
  --runs_root runs/eeuu_flatlogm --run_prefix ft_ \
  --tags l1base,l1sigma,l1oracle,l1base_s1,l1sigma_s1,l1oracle_s1,l1base_s2,l1sigma_s2,l1oracle_s2 \
  --out_prefix eeuu_reson_l1sd_ --summary eeuu_reson_l1seeds_summary.json
echo "DONE eval_l1_seeds"
