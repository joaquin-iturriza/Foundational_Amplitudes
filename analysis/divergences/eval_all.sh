#!/bin/bash
#SBATCH --job-name=ev_all
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=analysis/divergences/ev_all_%j.out
#SBATCH --error=analysis/divergences/ev_all_%j.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
# subprocess-isolated re-eval of EVERY arm (leak fix). raw=f0, flatlogm=resampled(fails),
# mix025/050/075=generated mixtures, genflat=f1, l1{base,sigma,oracle}=sigma-reweight.
python worktrees/wt-heterosc/analysis/divergences/eval_eeuu_resonance.py \
  --runs_root runs/eeuu_flatlogm --run_prefix ft_ \
  --tags raw,flatlogm,mix025,mix050,mix075,genflat,l1base,l1sigma,l1oracle \
  --out_prefix eeuu_reson_clean_ --summary eeuu_reson_clean_summary.json
echo "DONE eval_all"
