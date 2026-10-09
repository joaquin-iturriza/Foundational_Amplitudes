#!/bin/bash
#SBATCH --job-name=run_data_tools
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=run_data_tools_%j.out
#SBATCH --error=run_data_tools_%j.out
# tools/run_data_tools.py on a GPU node: RDT_ARGS="fisher --run-dir R` or `-- eval --run-dir R --weights W --out F`
# (continual-pretraining test, D21).
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
# site submit passes no arguments: they come in RDT_ARGS, words joined by '+' (the scheduler's environment export
# splits on spaces), e.g. site submit ... --env RDT_ARGS=fisher+--run-dir+runs/tp3_finale/trial_0073
if [ -n "${RDT_ARGS:-}" ]; then set -- ${RDT_ARGS//+/ }; fi
python tools/run_data_tools.py "$@"
