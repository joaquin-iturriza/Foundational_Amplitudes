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
    python tools/lhe_probe.py parts --queue ${PROBE_QUEUES:?} --out "$SCRATCH/lhe_probe/parts_${SLURM_JOB_ID}.json" ${PROBE_MATCH:+--match "$PROBE_MATCH"}
    echo "RUN_SUMMARY lhe_probe parts rc=$?"; exit 0
fi
if [ "$PROBE_SHARD" = mixcheck ]; then
    python tools/lhe_probe.py mixcheck --queue ${PROBE_QUEUES:?}
    python tools/lhe_probe.py runsize --queue ${PROBE_QUEUES:?}
    echo "RUN_SUMMARY lhe_probe mixcheck rc=$?"; exit 0
fi
if [ "$PROBE_SHARD" = gridpack ]; then
    # one gridpack of GP_PROCESS at the precision of a run of GP_GRID events (all cores), then GP_RUNS runs of GP_EVENTS
    # drawn from it with seeds GP_SEED_BASE+1.. (GP_ENERGY: an e+e- collider energy), kept under PROBE_KEEP/<process>/
    python tools/lhe_probe.py gridpack --process "${GP_PROCESS:?}" --grid-events "${GP_GRID:?}" --runs "${GP_RUNS:?}" \
        --events "${GP_EVENTS:?}" --seed-base "${GP_SEED_BASE:-0}" --cores "${SLURM_CPUS_PER_TASK:-1}" \
        ${GP_ENERGY:+--energy $GP_ENERGY} ${PROBE_KEEP:+--keep $PROBE_KEEP}
    echo "RUN_SUMMARY lhe_probe gridpack rc=$?"; exit 0
fi
if [ "$PROBE_SHARD" = gridpool ]; then
    # lhe.gridpack pools GP_POOLS (space separated), one after the other: each gridpack built at its path as the queue
    # builds it (all cores), then GP_RUNS probe runs of the pool's run size drawn from it, all cores at a time; a run-size
    # probe sets GP_EVENTS (run size), GP_ENERGY (one e+e- energy), GP_SEED_BASE and PROBE_KEEP (runs kept for sizecheck)
    python tools/lhe_probe.py gridpool --process ${GP_POOLS:?} --runs "${GP_RUNS:-16}" --cores "${SLURM_CPUS_PER_TASK:-1}" \
        ${GP_EVENTS:+--events $GP_EVENTS} ${GP_ENERGY:+--energy $GP_ENERGY} ${GP_SEED_BASE:+--seed-base $GP_SEED_BASE} \
        ${PROBE_KEEP:+--keep $PROBE_KEEP}
    echo "RUN_SUMMARY lhe_probe gridpool rc=$?"; exit 0
fi
if [ "$PROBE_SHARD" = list ]; then
    # explicit runs PROBE_LIST: "process,events,seed[,key=value...]" separated by spaces (a seed written s@E: the
    # e+e- collider energy E) (a run-card variant against
    # its control), each at most PROBE_TIMEOUT s; PROBE_KEEP=DIR keeps each run's events under DIR/<process>/
    T="${PROBE_TIMEOUT:-1500}"
    echo ${PROBE_LIST:?} | tr ' ' '\n' | tr ',' ' ' | xargs -L1 -P "${SLURM_CPUS_PER_TASK:-1}" sh -c \
        'p=$1 n=$2 s=${3%@*}; e=; case $3 in *@*) e="--energy ${3#*@}";; esac; shift 3; timeout '"$T"' python tools/lhe_probe.py run --process "$p" --events "$n" --seed "$s" $e --timeout '"$((T - 60))"' '"${PROBE_KEEP:+--keep $PROBE_KEEP}"' ${1:+--patch "$@"}' _
    echo "RUN_SUMMARY lhe_probe list rc=$?"; exit 0
fi
python tools/lhe_probe.py list --queue ${PROBE_QUEUES:?} | awk -v k="$PROBE_SHARD" -v n="${PROBE_NSHARDS:?}" 'NR % n == k' \
    | xargs -P "${SLURM_CPUS_PER_TASK:-1}" -I{} timeout 840 python tools/lhe_probe.py run --process {} \
        --events "${PROBE_EVENTS:-1000}" --timeout 780
echo "RUN_SUMMARY lhe_probe shard $PROBE_SHARD rc=$?"
