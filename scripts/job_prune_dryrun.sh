#!/bin/bash
# Print-only sweep/prune.py on the site it runs on (no --apply): what the prune would delete, into the job log.
# A CPU job because the cache glob over EOS outlasts `site run`'s ssh limit on lxplus.
#     site submit <site> FA scripts/job_prune_dryrun.sh --item <item> [-- <prune.py args>]   (default: --caches)
#SBATCH --job-name=prune_dryrun
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
case " $* " in *" --apply "*) echo "print-only: --apply refused"; exit 2;; esac
python sweep/prune.py "${@:---caches}"
RC=$?
echo "PRUNE_DRYRUN_DONE rc=$RC"
exit $RC
