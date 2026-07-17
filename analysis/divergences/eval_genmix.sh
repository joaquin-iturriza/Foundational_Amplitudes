#!/bin/bash
#SBATCH --job-name=ev_genmix
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=analysis/divergences/ev_genmix_%j.out
#SBATCH --error=analysis/divergences/ev_genmix_%j.out

set -e
module purge
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

# Full generated fraction sweep: f = 0 (raw) .. 1 (genflat). Each de-standardized to true
# log|M|^2, common RAMBO test binned by sqrt(s). Locates the interior optimum.
python worktrees/wt-heterosc/analysis/divergences/eval_eeuu_resonance.py \
  --runs_root runs/eeuu_flatlogm --run_prefix ft_ \
  --tags raw,mix025,mix050,mix075,genflat \
  --out_prefix eeuu_reson_sw_ --summary eeuu_reson_sweep_summary.json

echo "DONE eval_genmix"
