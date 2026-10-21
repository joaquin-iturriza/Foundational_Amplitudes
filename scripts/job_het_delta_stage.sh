#!/bin/bash
#SBATCH --job-name=het_delta_stage
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# D47/D71: a het_delta cell whose joint run blew up (run.py aborts, so the sigma-only stage of D70 never ran) still gets
# its value from its best checkpoint (CLAUDE.md, reported values): tools/sigma_stage.py runs the stage from that
# checkpoint, tools/eval_delta.py writes its per-event mu and sigma as job_het_delta.sh does. No retraining.
#     site submit <site> FA scripts/job_het_delta_stage.sh -- runs/het_delta/<sweep>__<trial> [...]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
rc=0
for OUT in "$@"; do
    OUT="${OUT%/}"
    python tools/sigma_stage.py "$OUT" || { rc=1; continue; }
    python tools/eval_delta.py --out-dir "$SCRATCH/zz_delta_het" "$OUT/sigma_stage" || rc=1
done
exit $rc
