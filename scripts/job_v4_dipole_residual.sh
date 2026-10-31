#!/bin/bash
#SBATCH --job-name=v4_dipole_residual
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=v4_dipole_residual_%j.out
#SBATCH --error=v4_dipole_residual_%j.out
#
# CPU only (plan v3_prep_explore, the dipole design): the per-event soft-dipole factor against the control pair's
# zero-shot reads already on the site ($SCRATCH/zz_v4_short_<arm>/{H2,TR}), momenta from the pools' frozen rows
# (analysis/catalog_v4/dipole_residual.py). One output per read directory under $SCRATCH/zz_v4_dipole_<job>; the
# summary lines are printed as DIPOLE_RESIDUAL {json}.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
# $1 = prop: the propagator-product read (log P; placebos E, logF, log D), the design of 2026-10-31
MODE="${1:-dipole}"; FLAG=(); TAG=DIPOLE_RESIDUAL
[ "$MODE" = prop ] && { FLAG=(--prop); TAG=PROP_RESIDUAL; }
# $1 = joint: the joint read (gradient-boosted fit of u on the event's kinematics); $2 $3 narrow it to one arm and one
# axis, so the four (arm, axis) reads go as four jobs in parallel
[ "$MODE" = joint ] && { FLAG=(--prop --joint); TAG=JOINT_RESIDUAL; }
# $1 = lcodump: the leave-class-out read's stage 1 (design of 2026-10-31): 500 events per probe of the leg-order-free
# features and u to $SCRATCH/zz_v4_lco_dump/<arm>_<axis>, the own-label ceiling on H2; $1 = lco: its stage 2
# (analysis/catalog_v4/lco_map.py over that dump, every arm). $4 = a probes file: the dry run, dumped to
# $SCRATCH/zz_v4_lco_dry instead (stage 2 on it: $1 = lco, $2 = zz_v4_lco_dry)
# $1 = lcodump_not / lco_not: the t-free re-read (2026-10-31; t, the scored event's own target, is a label): stage 1
# on the held-out pools only, the ceiling refitted without t, dumped to $SCRATCH/zz_v4_lco_dump_not; stage 2 with t
# excluded, its training pools from the first dump
NOT=(); [ "$MODE" = lcodump_not ] && { MODE=lcodump; NOT=(--exclude t); DUMPNOT=1; }
[ "$MODE" = lcodump ] && TAG=LCO_DUMP
DUMP=zz_v4_lco_dump; [ -n "${DUMPNOT:-}" ] && DUMP=zz_v4_lco_dump_not; [ -n "${4:-}" ] && DUMP=zz_v4_lco_dry
if [ "$MODE" = lco_not ]; then
    OUT="$SCRATCH/zz_v4_lco_not_${SLURM_JOB_ID:-$$}"; mkdir -p "$OUT"
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
    python analysis/catalog_v4/lco_map.py --root "$SCRATCH/zz_v4_lco_dump_not" --tr-root "$SCRATCH/zz_v4_lco_dump" --same-as "$SCRATCH/zz_v4_lco_dump" \
        --exclude t --out "$OUT/lco_map_not.json" || exit 1
    echo "V4_LCO_NOT_DONE $OUT"; exit 0
fi
if [ "$MODE" = lco ]; then
    OUT="$SCRATCH/zz_v4_lco_${SLURM_JOB_ID:-$$}"; mkdir -p "$OUT"
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
    python analysis/catalog_v4/lco_map.py --root "$SCRATCH/${2:-zz_v4_lco_dump}" --out "$OUT/lco_map.json" || exit 1
    echo "V4_LCO_DONE $OUT"; exit 0
fi
ARMS="${2:-v4_dimensionless_muon v4_dimensionless_muon_seed2}"; AXES="${3:-H2 TR}"
OUT="$SCRATCH/zz_v4_${MODE}_${SLURM_JOB_ID:-$$}"; mkdir -p "$OUT"
for arm in $ARMS; do
    for ax in $AXES; do
        d="$SCRATCH/zz_v4_short_$arm/$ax"
        [ -d "$d" ] || { echo "missing $d"; exit 1; }
        if [ "$MODE" = lcodump ]; then
            FLAG=(--prop --dump "$SCRATCH/$DUMP/${arm}_$ax"); [ "$ax" = H2 ] && FLAG+=(--ceiling "${NOT[@]}")
            [ -n "${4:-}" ] && FLAG+=(--probes "$4")
        fi
        s=$(python analysis/catalog_v4/dipole_residual.py "$d" "${FLAG[@]}" --out "$OUT/${arm}_$ax.json") || exit 1
        echo "$TAG {\"arm\": \"$arm\", \"axis\": \"$ax\", \"summary\": $s}"
    done
done
echo "V4_${TAG}_DONE $OUT"
