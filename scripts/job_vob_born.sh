#!/bin/bash
#SBATCH --job-name=vob_born
#SBATCH --cpus-per-task=6
#SBATCH --time=00:45:00
# Wall-time target 0.5 h: 6 pools in parallel, the largest 50000 rows of a 2->2 MadLoop point (estimate, at most
# 10 ms per point: 8 min), plus the standalone builds if they are not cached.
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
# the MadLoop standalones are not kept between generations (D208): build and pole-certify each once, serially
python -c "import datagen; [datagen.ensure_virt_backend(p) for p in ('ee_uu_nlo', 'ee_ttbar_nlo')]" || exit 1
pids=()
for spec in "ee_uu ee_uu_nlo_25-1000GeV" "ee_ttbar ee_ttbar_nlo_362-1000GeV"; do
  set -- $spec
  for role in train val test; do
    d=$FROZEN; [ "$role" = train ] && d=$CACHE
    pool="$d/${2}_${role}_amplitudes.npy"
    python tools/nlo_virtual_pipeline.py "$1" --born-of "$pool" --n 100000000 --out "$OUT/${2}_${role}.npz" \
      > "$OUT/${2}_${role}.log" 2>&1 &
    pids+=($!)
  done
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
grep -h "^BORN_OF_POOL" "$OUT"/*.log
[ $rc = 0 ] || { tail -n 30 "$OUT"/*.log; exit 1; }
echo "VOB_BORN_DONE"
