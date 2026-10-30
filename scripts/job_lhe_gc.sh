#!/bin/bash
#SBATCH --job-name=lhe_gc
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=lhe_gc_%j.out
#SBATCH --error=lhe_gc_%j.out
#
# CPU only. Tidies the MadEvent working directories of the queues' runs no live job holds (tools/lhe_parts.py gc:
# a saved run's whole directory, any other's all but its logs; manifest under each queue's gc/ first), then any
# directory older than 12 h left on $SCRATCH/lhe_work (tools/lhe_parts.py workdirs --apply).
#     site submit ccin2p3 FA scripts/job_lhe_gc.sh --env LHE_QUEUES="<name> <name>" --item <entry>
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
df -B1G "$SCRATCH" | tail -1
rc=0
for q in ${LHE_QUEUES:-}; do
    python tools/lhe_parts.py gc --queue "$SCRATCH/lhe_queue/$q" || rc=1
done
df -B1G "$SCRATCH" | tail -1
# every MadEvent directory left on the shared tree by no queue's current layout, or by a failed series build
python tools/lhe_parts.py workdirs --apply || rc=1
du -sh "$SCRATCH/lhe_work"
echo "RUN_SUMMARY lhe_gc rc=$rc"
exit $rc
