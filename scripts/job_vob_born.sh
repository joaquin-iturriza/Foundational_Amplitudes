#!/bin/bash
#SBATCH --job-name=vob_born
#SBATCH --cpus-per-task=6
#SBATCH --time=01:15:00
# Wall-time target: 6 pools in parallel, the largest 50000 rows of a 2->2 MadLoop point (estimate, at most
# 10 ms per point: 8 min), plus the two standalone builds and pole checks (estimate 10-20 min each, serial): 0.8 h.
# With arguments (plan pretrain_data_saving): one-loop processes as <mg process>:<lo>-<hi> (mg5_pipeline_final names,
# e.g. v4_bg_ab_nlo:20-6000), each built and certified in parallel, then its three pools read in parallel; set
# --cpus-per-task to 3 x the processes at submit.
#SBATCH --output=vob_born_%j.out
#SBATCH --error=vob_born_%j.out
#
# CPU only (plan virtual_over_born): MadLoop's Born re-evaluated on every row of the one-loop e+e- -> u u~ and
# e+e- -> t t~ pools (train, val, test; tools/nlo_virtual_pipeline.py --born-of, one process per pool: the MadLoop
# module is a per-process singleton), the stored target recomputed and checked against the stored column; a pool whose
# every row reproduces gets its row-aligned Born sidecar <pool>.born.npy, which the V/B target reads. The npz of each
# pool goes under $SCRATCH/zz_vob_born_<job>.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SETUPTOOLS_USE_DISTUTILS=stdlib
cd "$PROJECT_DIR"
CACHE=$(python -c "import datagen; print(datagen.train_cache_dir())")
FROZEN=$(python -c "import datagen; print(datagen.frozen_dir())")
OUT="$SCRATCH/zz_vob_born_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
SPECS=("$@"); [ ${#SPECS[@]} -gt 0 ] || SPECS=(ee_uu_nlo:25-1000 ee_ttbar_nlo:362-1000)
PROCS=(); for s in "${SPECS[@]}"; do PROCS+=("${s%%:*}"); done
# the MadLoop standalones are not kept between generations (D208): build and pole-certify each once; distinct
# standalones share no state (datagen.ensure_backends), so they build in parallel
bpids=()
for p in "${PROCS[@]}"; do
  python -c "import datagen; datagen.ensure_virt_backend('$p')" > "$OUT/build_$p.log" 2>&1 &
  bpids+=($!)
done
for b in "${bpids[@]}"; do wait "$b" || { tail -n 30 "$OUT"/build_*.log; exit 1; }; done
pids=()
for s in "${SPECS[@]}"; do
  p="${s%%:*}"; w="${s#*:}"
  base=$(python -c "import mg5_pipeline_final as mg; print(mg.PROCESSES['$p']['virt_base'])") || exit 1
  for role in train val test; do
    d=$FROZEN; [ "$role" = train ] && d=$CACHE
    pool="$d/${p}_${w}GeV_${role}_amplitudes.npy"
    [ -e "$pool" ] || { echo "no pool $pool"; exit 1; }
    python tools/nlo_virtual_pipeline.py "$base" --born-of "$pool" --n 100000000 --out "$OUT/${p}_${w}GeV_${role}.npz" \
      > "$OUT/${p}_${role}.log" 2>&1 &
    pids+=($!)
  done
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
grep -h "^BORN_OF_POOL" "$OUT"/*.log
[ $rc = 0 ] || { tail -n 30 "$OUT"/*.log; exit 1; }
echo "VOB_BORN_DONE"
