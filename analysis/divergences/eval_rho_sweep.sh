#!/bin/bash
#SBATCH --job-name=q2rhoev
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=analysis/divergences/q2rhoev_%j.out
#SBATCH --error=analysis/divergences/q2rhoev_%j.out
set -e
module purge; module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
python analysis/divergences/eval_heldout.py \
  --runs_root runs/q2_sigma_reweight \
  --run_prefix ft_ --tags rho10,rho20,rho30,rho40,rho50,rho60,rho70,rho80,rho90 \
  --heldout analysis/divergences/uug_deep_test.npz \
  --out_prefix q2rw_eval_ \
  --summary q2rho_eval_summary.json
echo "DONE q2 rho eval"
