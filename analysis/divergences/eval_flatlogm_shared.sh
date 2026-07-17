#!/bin/bash
#SBATCH --job-name=ev_flatlogm_sh
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=analysis/divergences/ev_flatlogm_sh_%j.out
#SBATCH --error=analysis/divergences/ev_flatlogm_sh_%j.out

set -e
module purge
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

# Fair A/B eval: de-standardize BOTH arms with the shared native-pool stats.
python worktrees/wt-heterosc/analysis/divergences/eval_eeuu_resonance.py \
  --runs_root runs/eeuu_flatlogm_shared --run_prefix ft_ --tags raw,flatlogm \
  --frozen_stats analysis/divergences/eeuu_shared_stats.json \
  --out_prefix eeuu_reson_sh_ --summary eeuu_reson_shared_summary.json

echo "DONE eval_flatlogm_shared"
