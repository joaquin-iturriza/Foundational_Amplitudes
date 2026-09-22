#!/bin/bash
#SBATCH --job-name=evalho
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=analysis/divergences/evalho_%j.out
#SBATCH --error=analysis/divergences/evalho_%j.out
#SBATCH --gres=gpu:1
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"/worktrees/wt-l2-uugg
export PYTHONDONTWRITEBYTECODE=1
# args: space-separated "run_dir:label" pairs via $RUNS
for spec in $RUNS; do
  rd="${spec%%:*}"; lb="${spec##*:}"
  python analysis/divergences/eval_heldout_uugg.py --run_dir "$rd" --label "$lb" || echo "EVAL FAILED: $rd"
done
echo "DONE evalho"
