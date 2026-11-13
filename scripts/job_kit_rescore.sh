#!/bin/bash
#SBATCH --job-name=kit_rescore
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# kit_benchmark: re-score finished KIT runs with tools/kit_predict.py (each event read in its own frame, unaugmented:
# the prediction depends on the event alone). The earlier scores, whose augmentation was keyed on the scored pool,
# are kept once in <run>/kit_benchmark_poolkeyed for the comparison. No training.
#     site submit ccin2p3 FA scripts/job_kit_rescore.sh --item kit_benchmark -- <run_dir> ...
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
RC=0
for R in "$@"; do
    # an unscored run (a kit_hpo trial, scored as a design control) has no earlier scores to keep
    [ -d "$R/kit_benchmark_poolkeyed" ] || [ ! -d "$R/kit_benchmark" ] || cp -a "$R/kit_benchmark" "$R/kit_benchmark_poolkeyed" || { RC=1; continue; }
    python tools/kit_predict.py "$R" > /dev/null && echo "RESCORED $R" || { echo "RESCORE_FAILED $R"; RC=1; }
done
[ "$RC" -eq 0 ] && echo "RUN_DONE"
exit $RC
