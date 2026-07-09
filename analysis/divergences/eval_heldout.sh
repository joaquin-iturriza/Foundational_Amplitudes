#!/bin/bash
#SBATCH --job-name=ho_eval
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=analysis/divergences/ho_eval_%j.out
#SBATCH --error=analysis/divergences/ho_eval_%j.out

set -e
module purge
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

# Forward each add-back fine-tune over the fixed held-out NEAR (deep-IR) region and
# compute RMS Δlog|M|^2 inside it. Fine-tune best ckpts are gzipped (.pt.gz).
python analysis/divergences/eval_heldout.py \
  --tags 000,005,015,050,100 \
  --ckpt model_run0_best.pt.gz

echo "DONE ho_eval"
