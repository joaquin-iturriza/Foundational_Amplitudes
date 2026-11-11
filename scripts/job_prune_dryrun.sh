#!/bin/bash
# Print-only cache pass of sweep/prune.py (prune.caches, no deletion) on the site it runs on: one CACHE_DRY line per
# file an --apply would delete, into the job log. A CPU job because the cache glob over EOS outlasts `site run`'s ssh
# limit on lxplus; the cache pass alone, because prune.py's main queries the scheduler first (condor_q fails on an
# lxplus worker node).
#     site submit <site> FA scripts/job_prune_dryrun.sh --item <item> [-- <cache days, default 30>]
#SBATCH --job-name=prune_dryrun
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
python -c "import sys; sys.path.insert(0, 'sweep'); import prune; print('SUMMARY caches dry run: %.2f GB' % (prune.caches(float(sys.argv[1]), False) / 1e9))" "${1:-30}"
RC=$?
echo "PRUNE_DRYRUN_DONE rc=$RC"
exit $RC
