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
if [ "$PROBE_SHARD" = reread ]; then
    # par:v4_short_runs re-read (catalog_coverage): each committed measurement re-run on the files it saw (--before its
    # commit time) at 20000 relabellings, with the old unscaled run-level test and the scaled one (same relabellings, so
    # paired), all in parallel; then committed against unscaled (event KS must reproduce) and unscaled against scaled
    R="$SCRATCH/reread"; mkdir -p "$R"; D=analysis/catalog_v4/data; P=analysis/catalog_v4/short_run_bias.py; N=20000
    for v in unscaled scaled; do
        F=; [ $v = unscaled ] && F=--unscaled
        python $P measure --nperm $N $F --before 2026-10-30T05:38:03+01:00 --out "$R/$v/short_run_bias.json" > "$R/$v.m1.log" 2>&1 &
        python $P measure --nperm $N $F --before 2026-10-30T05:59:00+01:00 --out "$R/$v/short_run_bias_size.json" > "$R/$v.m2.log" 2>&1 &
        python $P measure --nperm $N $F --before 2026-10-30T06:30:34+01:00 --out "$R/$v/short_run_bias_gpack.json" > "$R/$v.m3.log" 2>&1 &
        python $P sizecheck --nperm $N $F --process ee4_epem_cgsxwm --energy 3000 --keep "$SCRATCH/lhe_probe_keep_runsize" --ref-ask 200 --out "$R/$v/cgsxwm_sizecheck.json" > "$R/$v.m4.log" 2>&1 &
    done
    wait
    tail -n 3 "$R"/*.log
    for f in short_run_bias short_run_bias_size short_run_bias_gpack cgsxwm_sizecheck; do
        echo "== $f committed vs unscaled"; python $P reread --old "$D/$f.json" --new "$R/unscaled/$f.json"
        echo "== $f unscaled vs scaled"; python $P reread --old "$R/unscaled/$f.json" --new "$R/scaled/$f.json"
    done
    echo "RUN_SUMMARY lhe_probe reread rc=$?"; exit 0
fi
if [ "$PROBE_SHARD" = reread2 ]; then
    # the re-read again on the files each measurement saw (catalog_coverage step 13): the commit times of reread were
    # later than the measurements and set-aside pools were missed; cutoff finds, per measurement, the earliest file time
    # within 12 h before its commit at which the most groups' runs and events match, then measure there (unscaled and
    # scaled, 20000 relabellings, paired) and reread against the committed numbers
    R="$SCRATCH/reread2"; mkdir -p "$R"; D=analysis/catalog_v4/data; P=analysis/catalog_v4/short_run_bias.py; N=20000
    M=(short_run_bias short_run_bias_size short_run_bias_gpack)
    python $P cutoff --old $D/short_run_bias.json $D/short_run_bias_size.json $D/short_run_bias_gpack.json \
        --before 2026-10-30T05:38:03+01:00 2026-10-30T05:59:00+01:00 2026-10-30T06:30:34+01:00 --hours 12 \
        --key-v1 $D/short_run_bias.json $D/short_run_bias_size.json | tee "$R/cutoff.log"
    mapfile -t T < <(python -c "import json,sys; [print(json.loads(l[12:])['best_time']) for l in open(sys.argv[1]) if l.startswith('CUTOFF_JSON ')]" "$R/cutoff.log")
    [ "${#T[@]}" = 3 ] || { echo "RUN_SUMMARY lhe_probe reread2 rc=1 (cutoff gave ${#T[@]} times)"; exit 1; }
    for t in "${T[@]}"; do [ "$t" != None ] || { echo "RUN_SUMMARY lhe_probe reread2 rc=1 (a measurement matched at no time)"; exit 1; }; done
    K=(--key-v1 --key-v1 "")    # short_run_bias and _size were measured before 7bdc5729 changed the physics key
    for v in unscaled scaled; do
        F=; [ $v = unscaled ] && F=--unscaled
        for i in 0 1 2; do
            python $P measure --nperm $N $F ${K[$i]} --before "${T[$i]}" --out "$R/$v/${M[$i]}.json" > "$R/$v.${M[$i]}.log" 2>&1 &
        done
    done
    wait
    tail -n 1 "$R"/*.log
    for f in "${M[@]}"; do
        echo "== $f committed vs unscaled"; python $P reread --old "$D/$f.json" --new "$R/unscaled/$f.json"
        echo "== $f unscaled vs scaled"; python $P reread --old "$R/unscaled/$f.json" --new "$R/scaled/$f.json"
    done
    echo "RUN_SUMMARY lhe_probe reread2 rc=$?"; exit 0
fi
if [ "$PROBE_SHARD" = ppcheck4 ]; then
    # the pp13v4 validation (catalog_coverage step 10): every pp13v4 entry's three role pools, then catalog_v4's
    # standalone on the first 200 rows of each val pool against the stored label, 16 parts in parallel, and the summary
    R="$SCRATCH/ppcheck4"; mkdir -p "$R"; P=analysis/catalog_v4/pp_physics_check.py
    python $P built --set v4 | tee "$R/built.json"; echo
    # the e+e- default test pools' diagram sidecars (the EE4 read's order features; none built yet), alongside
    python tools/dump_diagrams.py --prefix ee4_ --skip-existing --workers 16 > "$R/ee4_diagrams.log" 2>&1 &
    for k in $(seq 0 15); do python $P eval --set v4 --n 200 --part $k --parts 16 > "$R/eval_$k.json" 2> "$R/eval_$k.log" & done
    wait
    python $P summary --set v4 --evals "$R/eval_*.json" --out "$R/pp13v4_physics_check.json" | sed 's/^/PPCHECK4_SUMMARY /'
    tail -n 2 "$R/ee4_diagrams.log"; python tools/check_site_data.py recipes/catalog_v4_ee_test.yaml | sed 's/^/EE4_DATA /'
    echo "RUN_SUMMARY lhe_probe ppcheck4 rc=$?"; exit 0
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
        ${PROBE_KEEP:+--keep $PROBE_KEEP} ${GP_KEEP_LOGS:+--keep-logs $GP_KEEP_LOGS} ${GP_TIMEOUT:+--timeout $GP_TIMEOUT} \
        ${GP_XIMPROVE:+--ximprove $GP_XIMPROVE}
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
