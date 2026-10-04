#!/bin/bash
#SBATCH --job-name=ew_sudakov_check
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=ew_sudakov_check_%j.out
#SBATCH --error=ew_sudakov_check_%j.out
#
# CPU only: does a MadLoop [virt=QED] e+e- -> d d~ target carry log^2(s/M_W^2)? (tools/ew_sudakov_check.py)

_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
python tools/ew_sudakov_check.py
cat analysis/transfer/ew_sudakov_check.json | python -c "import json,sys; d=json.load(sys.stdin); print('FITS', json.dumps({k: v.get('fits', v) for k, v in d.items()}))"
