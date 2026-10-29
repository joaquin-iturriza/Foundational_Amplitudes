#!/bin/bash
#SBATCH --job-name=lhe_gc
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=lhe_gc_%j.out
#SBATCH --error=lhe_gc_%j.out
#
# CPU only. Tidies the MadEvent working directories of the queues' runs no live job holds (tools/lhe_parts.py gc:
# a saved run's whole directory, any other's all but its logs; manifest under each queue's gc/ first).
#     site submit ccin2p3 FA scripts/job_lhe_gc.sh --env LHE_QUEUES="<name> <name>" --item <entry>
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
df -B1G "$SCRATCH" | tail -1
rc=0
for q in ${LHE_QUEUES:?LHE_QUEUES (queue names under \$SCRATCH/lhe_queue) not set}; do
    python tools/lhe_parts.py gc --queue "$SCRATCH/lhe_queue/$q" || rc=1
done
df -B1G "$SCRATCH" | tail -1
du -sh "$SCRATCH/lhe_work/parts"
echo "RUN_SUMMARY lhe_gc rc=$rc"
exit $rc
