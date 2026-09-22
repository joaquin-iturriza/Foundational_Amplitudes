#!/bin/bash
# Zero-shot eval of a frozen pretrained model on the two held-out processes.
# Usage: sbatch tools/run_zero_shot.sh <ckpt_run_dir> <out_label>
#   e.g. sbatch tools/run_zero_shot.sh runs/pretrain25/trial_0009 phys25_500k
#SBATCH --job-name=zshot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=zshot_%j.out
#SBATCH --error=zshot_%j.err
#SBATCH --gres=gpu:1

set -uo pipefail
PY=$PROJECT_DIR/.venv/bin/python
cd $PROJECT_DIR

RUN_DIR="${1:?usage: sbatch run_zero_shot.sh <ckpt_run_dir> <out_label>}"
LABEL="${2:?need out_label}"
OUT=analysis/zero_shot
# Defaults: recipe-generated held-out test pools (same pipeline/convention as
# training). Override DATA_PATH/DATASETS for the old files-based datasets.
DATA_PATH="${DATA_PATH:-${DATA_DIR:?DATA_DIR is set by sites/activate.sh}}"
DATASETS="${DATASETS:-ee_uu_10-1000GeV_test_amplitudes ee_ttbar_346-1000GeV_test_amplitudes}"

for DS in $DATASETS; do
  echo "=== zero-shot: $LABEL on $DS ==="
  $PY tools/zero_shot_eval.py \
    --ckpt-run-dir "$RUN_DIR" \
    --dataset "$DS" \
    --data-path "$DATA_PATH" \
    --out "$OUT/${LABEL}__${DS}.json" \
    --subsample 30000
done
echo "ALL DONE $LABEL"
