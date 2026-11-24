#!/bin/bash
#SBATCH --job-name=llm_score
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=llm_score_%j.out
#SBATCH --error=llm_score_%j.out
#
# llm_compare's test read of one trained run (tools/llm_score.py): pooled RMSE of log|M|^2 on the test pools of its own
# training recipe (seen) and of the held-out recipe (the run's shared statistics), into <run>/llm_score_{seen,heldout}.json.
#   site submit <site> FA scripts/job_llm_score.sh --env RUN=<run dir> [--env HELD=<held-out recipe>] --item llm_compare
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
RUN=${1:-$RUN}; HELD=${2:-${HELD:-recipes/catalog_daniel_heldout.yaml}}   # args, or --env RUN=/HELD= (site submit passes no args)
SEEN=$(python -c "import sys; sys.path.insert(0, 'tools'); from rebuild_run import load_cfg; print(load_cfg('$RUN').data.processes_file)") || exit 1
python tools/llm_score.py --run "$RUN" --out "$RUN/llm_score_seen.json" "$SEEN" || exit 1
python tools/llm_score.py --run "$RUN" --out "$RUN/llm_score_heldout.json" "$HELD" || exit 1
echo "RUN_DONE"
