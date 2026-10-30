#!/bin/bash
#SBATCH --job-name=lhe_probe
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=lhe_probe_%j.out
#SBATCH --error=lhe_probe_%j.out
#
# CPU only. The lhe part work's probe (tools/lhe_probe.py; catalog_coverage step 7): the node's disks, then one short
# MadEvent run (PROBE_EVENTS, 13 min at most) of each process of shard PROBE_SHARD of PROBE_NSHARDS of the processes not
# assembled in the queues PROBE_QUEUES, one per core, in node-local $TMPDIR. PROBE_SHARD=parts instead reads the part
# files already made (yields, seconds, the fraction a pool keeps) into $SCRATCH/lhe_probe/parts_<job>.json.
#     site submit ccin2p3 FA scripts/job_lhe_probe.sh --env PROBE_QUEUES="q1 q2" --env PROBE_SHARD=k --env PROBE_NSHARDS=n --item <entry>
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
python tools/lhe_probe.py node
if [ "${PROBE_SHARD:?}" = parts ]; then
    mkdir -p "$SCRATCH/lhe_probe"
    python tools/lhe_probe.py parts --queue ${PROBE_QUEUES:?} --out "$SCRATCH/lhe_probe/parts_${SLURM_JOB_ID}.json"
    echo "RUN_SUMMARY lhe_probe parts rc=$?"; exit 0
fi
python tools/lhe_probe.py list --queue ${PROBE_QUEUES:?} | awk -v k="$PROBE_SHARD" -v n="${PROBE_NSHARDS:?}" 'NR % n == k' \
    | xargs -P "${SLURM_CPUS_PER_TASK:-1}" -I{} timeout 840 python tools/lhe_probe.py run --process {} \
        --events "${PROBE_EVENTS:-1000}" --timeout 780
echo "RUN_SUMMARY lhe_probe shard $PROBE_SHARD rc=$?"
