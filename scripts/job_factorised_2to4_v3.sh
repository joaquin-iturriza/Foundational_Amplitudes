#!/bin/bash
#SBATCH --job-name=factorised_v3
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=factorised_v3_%j.out
#SBATCH --error=factorised_v3_%j.out
#
# CPU only: the factorised 2->4 head's data check on catalog_v3 (the level item, moved to catalog_v3 by the user's call
# of 2026-10-25): its 2->4 classes built as a 2->3 class plus a gluon, photon, Z or H, up to 40 per kind (seed 0), each
# against its parent's exact |M|^2 at the mapped events (tools/factorised_2to4.py --v3); $SCRATCH/zz_factorised_v3/
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
python tools/factorised_2to4.py --v3 40 --seed 0 --n 2000 --out "$SCRATCH/zz_factorised_v3"
