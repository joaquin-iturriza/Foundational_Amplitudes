#!/bin/bash
#SBATCH --job-name=lhe_parts
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=lhe_parts_%j.out
#SBATCH --error=lhe_parts_%j.out
#
# CPU only. One worker of a queue of one-run MadEvent part jobs (tools/lhe_parts.py; catalog_coverage step 7): it
# drains $SCRATCH/lhe_queue/$LHE_QUEUE on its cores, longest task first, never claiming a task that would not end
# inside its time limit, and assembles a process's pools when its part completes the process. Any number of these
# run on one queue. The queue is planned beforehand (site run ... python tools/lhe_parts.py plan SPEC... --queue Q).
#     site submit ccin2p3 FA scripts/job_lhe_parts.sh --env LHE_QUEUE=<name> [--env LHE_HOURS=<h>] --item <entry>
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
Q="$SCRATCH/lhe_queue/${LHE_QUEUE:?LHE_QUEUE (the queue name under \$SCRATCH/lhe_queue) not set}"
export AMP_LHE_WORK_DIR="${AMP_LHE_WORK_DIR:-$SCRATCH/lhe_work}"
# the time limit less 5 min (claims stop when the longest remaining fit is past it)
H="${LHE_HOURS:-$(squeue -h -j "$SLURM_JOB_ID" -o %l | awk -F'[-:]' '{ if (NF==4) s=$1*86400+$2*3600+$3*60+$4; else if (NF==3) s=$1*3600+$2*60+$3; else s=$1*60+$2; printf "%.3f", (s-300)/3600 }')}"
echo "queue $Q  cores ${SLURM_CPUS_PER_TASK:-1}  hours $H"
python tools/lhe_parts.py status --queue "$Q"
python tools/lhe_parts.py work --queue "$Q" --workers "${SLURM_CPUS_PER_TASK:-1}" --hours "$H"
rc=$?
python tools/lhe_parts.py status --queue "$Q"
echo "RUN_SUMMARY lhe_parts rc=$rc"
exit $rc
