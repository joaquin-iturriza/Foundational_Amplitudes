#!/bin/bash
#SBATCH --job-name=restore_config
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=restore_config_%j.out
#SBATCH --error=restore_config_%j.out
#
# Rebuilds a run's overwritten config.yaml from its seed twin (tools/restore_run_config.py; the damaged file is kept
# as config.yaml.overwritten_<n>). Arguments: <run_dir> <twin_dir> relative to the checkout, then the tool's options.
#   site submit ccin2p3 FA scripts/job_restore_config.sh --env RESTORE_ARGS="runs/a runs/b" --item <item>
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
# shellcheck disable=SC2086
python tools/restore_run_config.py ${RESTORE_ARGS:?RESTORE_ARGS="<run_dir> <twin_dir> [options]"} "$@"; rc=$?
[ "$rc" = 0 ] && echo "RUN_DONE"
exit $rc
