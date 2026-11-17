#!/bin/bash
#SBATCH --job-name=zng_zeroshot
#SBATCH --cpus-per-task=5
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=zng_zeroshot_%j.out
#SBATCH --error=zng_zeroshot_%j.out
#
# zng_uq_benchmark step 3: the catalog_v4 pretraining (trial 1262, its sigma stage: mu frozen, sigma calibrated, D70)
# zero-shot on the UQ repo's q qbar -> Z+1..4g test events (Zenodo 16793011, their split; no Z+ng in its catalog),
# each event's prediction kept (tools/zero_shot_parent.py --save-dir; t, q, s in ln|M|^2), read by
# analysis/zng_uq/benchmark.py collect.
#   site submit ccin2p3 FA scripts/job_zng_zeroshot.sh --item zng_uq_benchmark -- [n test events, default 1000000]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
NTEST="${1:-1000000}"
P="${ZNG_PARENT:-runs/catalog_v4_long/best_cov_v4_long_hpo8_cc_68224_trial_1262/sigma_stage}"
OUT="runs/kit/zng_zeroshot_1262_n${NTEST}"
R="$OUT/recipes"; mkdir -p "$R" "$OUT/preds"
for p in zg zgg zggg zgggg; do
    printf 'processes:\n- name: kit_uq_%s__me2\n  sqrts: [1000, 1000]\n  n_train: 500000\n  n_val: 100000\n  n_test: %s\n' \
        "$p" "$NTEST" > "$R/kit_uq_${p}__me2.yaml"
done
python tools/zero_shot_parent.py --parent "$P" --split test --eval-subsample "$NTEST" --save-dir "$OUT/preds" \
    --allow-physics-mismatch "Zenodo 16793011 events: |M|^2 equals our MadGraph d d~ > Z+ng to 1.4e-14 (FA-ccin2p3-32145d4b)" \
    "$R"/kit_uq_*.yaml
echo "ZNG_ZEROSHOT_DONE $OUT"
