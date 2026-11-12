#!/bin/bash
#SBATCH --job-name=kit_mlp
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# kit_benchmark, D403: Bakar's reference MLP and its one-change ladder towards our setup (tools/kit_mlp_ladder.py),
# every arm x seed of one KIT process; then, with --baseline, his own numpy src/baseline.py once (the port's check).
#     site submit ccin2p3 FA scripts/job_kit_mlp_ladder.sh --item kit_benchmark -- <file> <target> [--baseline]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
F="$1"; T="$2"; RC=0
for ARM in mlp target mom raw gelu het cur recipe ours; do
    for S in 0 1 2; do
        python tools/kit_mlp_ladder.py "$F" "$T" --arm "$ARM" --seed "$S" | grep MLP_DONE || { echo "MLP_FAILED $ARM $S"; RC=1; }
    done
done
if [ "$3" = --baseline ]; then
    B="$DATA_DIR/kit_hardps/hardps_benchmark"; O="runs/kit_mlp/baseline_${F}_${T}"; mkdir -p "$O"
    FEAT=log; [ "$F" = ee_tt ] && FEAT=logthr
    TG="$T"; [ "$F" = ee_tt ] && TG="abs_$T"
    python -I "$B/src/baseline.py" "$B/data/full/$F.h5" "$TG" "$FEAT" "$O/predictions.h5" \
        && python -I "$B/src/evaluate.py" "$B/data/full/$F.h5" "$O/predictions.h5" "$O/kit_result.json" > "$O/kit_result.txt" \
        && echo "BASELINE_DONE $O" || { echo "BASELINE_FAILED"; RC=1; }
fi
[ "$RC" -eq 0 ] && echo "RUN_DONE"
exit $RC
