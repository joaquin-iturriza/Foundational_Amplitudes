#!/bin/bash
#SBATCH --job-name=kit_numpy
#SBATCH --cpus-per-task=12
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# kit_benchmark, D403 port gate: his numpy src/baseline.py at seeds 0 1 2 (tools/kit_baseline_seeded.py), each scored by
# his evaluate.py, in parallel on CPU (4 threads each). runs/kit_mlp/baseline_<file>_<target>_s<seed>.
#     site submit ccin2p3 FA scripts/job_kit_numpy_seeds.sh --item kit_benchmark -- <file> <target>
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
F="$1"; T="$2"; B="$DATA_DIR/kit_hardps/hardps_benchmark"
FEAT=log; [ "$F" = ee_tt ] && FEAT=logthr
TG="$T"; [ "$F" = ee_tt ] && TG="abs_$T"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for S in 0 1 2; do
    O="runs/kit_mlp/baseline_${F}_${T}_s$S"; mkdir -p "$O"
    ( python tools/kit_baseline_seeded.py "$S" "$B/src/baseline.py" "$B/data/full/$F.h5" "$TG" "$FEAT" "$O/predictions.h5" \
      && python -I "$B/src/evaluate.py" "$B/data/full/$F.h5" "$O/predictions.h5" "$O/kit_result.json" > "$O/kit_result.txt" \
      && echo "BASELINE_DONE $O" || echo "BASELINE_FAILED $S" ) &
done
wait
[ "$(ls runs/kit_mlp/baseline_${F}_${T}_s*/kit_result.json 2>/dev/null | wc -l)" -eq 3 ] && echo "RUN_DONE"
