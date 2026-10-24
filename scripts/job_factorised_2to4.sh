#!/bin/bash
#SBATCH --job-name=factorised_2to4
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=factorised_2to4_%j.out
#SBATCH --error=factorised_2to4_%j.out
#
# CPU only: the level item's design (ii) checked on the data, the 40 2->4 probes against their 2->3 parents' exact
# |M|^2 at the mapped events (tools/factorised_2to4.py); outputs in $SCRATCH/zz_factorised_2to4/
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
python tools/factorised_2to4.py --n 2000 "$@"
