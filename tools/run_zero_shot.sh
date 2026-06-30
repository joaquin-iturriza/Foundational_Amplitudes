#!/bin/bash
# Zero-shot eval of a frozen pretrained model on the two held-out processes.
# Usage: sbatch tools/run_zero_shot.sh <ckpt_run_dir> <out_label>
#   e.g. sbatch tools/run_zero_shot.sh runs/pretrain25/trial_0009 phys25_500k
#SBATCH --job-name=zshot
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=zshot_%j.out
#SBATCH --error=zshot_%j.err

set -euo pipefail
module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

RUN_DIR="${1:?usage: sbatch run_zero_shot.sh <ckpt_run_dir> <out_label>}"
LABEL="${2:?need out_label}"
OUT=analysis/zero_shot

for DS in ee_uu_91-1000GeV_amplitudes ee_ttbar_346-1000GeV_amplitudes; do
  echo "=== zero-shot: $LABEL on $DS ==="
  python tools/zero_shot_eval.py \
    --ckpt-run-dir "$RUN_DIR" \
    --dataset "$DS" \
    --out "$OUT/${LABEL}__${DS}.json" \
    --subsample 30000
done
echo "ALL DONE $LABEL"
