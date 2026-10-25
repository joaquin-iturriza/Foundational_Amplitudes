#!/bin/bash
#SBATCH --job-name=lowd_ft
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=lowd_ft_%j.out
#SBATCH --error=lowd_ft_%j.out
#
# The small-D fine-tuning study, first round (D120): one method on one probe's 12 cells (rungs 1, 7, 9 at D = 10 to
# 10^2.5), tools/lowd_ft.py; or, with method "readout", the frozen linear read-out of all 60 cells (tools/linear_readout.py).
#     site submit <site> FA scripts/job_lowd_ft.sh -- <default|lpft|l2sp|freeze|readout> [probe]   (FA_ONLY="r7_d3" for a smoke)
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
M="$1"; P="$2"
if [ "$M" = readout ]; then
  CFGS=$(python -c "import json; print(' '.join(sorted({c['config'] for c in json.load(open('analysis/transfer/data/lowd_ft_spec.json'))})))")
  python tools/linear_readout.py --out "$SCRATCH/zz_lowd/readout" $CFGS
else
  # FA_ONLY="r7_d3 ..." restricts to those cells (the smoke test); for a smoke, several methods: M="lpft l2sp freeze"
  for m in $M; do python tools/lowd_ft.py --method "$m" --probe "$P" --out "$SCRATCH/zz_lowd${FA_ONLY:+_smoke}" ${FA_ONLY:+--only $FA_ONLY}; done
fi
