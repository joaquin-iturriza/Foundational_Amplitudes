#!/bin/bash
#SBATCH --job-name=catalog_ft
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=catalog_ft_%j.out
#SBATCH --error=catalog_ft_%j.out
#
# The Muon test's small-D fine-tunes (D144, D146): tools/catalog_ft.py from one catalog_v3 short pretraining with one
# fine-tune optimizer, the 4 probes x 4 D cells; with "scratch", the same cells from a fresh initialisation (the parent's
# config and HPs, no weights): whether fine-tuning a current pretraining beats scratch at small D (the user, 2026-10-26);
# with "keep-parent-stats", the fine-tunes keep the parent's target standardization (step 0 = the parent's zero-shot).
#     site submit <site> FA scripts/job_catalog_ft.sh --item muon_test -- <parent arm> <AdamW|MuonAdamW> [scratch|keep-parent-stats|-] [out dir under $SCRATCH, default zz_catft]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
ARM="${1:?parent arm under runs/catalog_v3_short}"; OPT="${2:?AdamW or MuonAdamW}"; MODE="${3:-}"; [ "$MODE" = "-" ] && MODE=""; OUT="${4:-zz_catft}"
python tools/catalog_ft.py --parent "runs/catalog_v3_short/$ARM" --optimizer "$OPT" --out "$SCRATCH/$OUT" \
    ${MODE:+--$MODE} || exit 1
echo "CATFT_DONE $ARM $OPT $MODE"
