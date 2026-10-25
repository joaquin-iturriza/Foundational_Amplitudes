#!/bin/bash
#SBATCH --job-name=factorised_v3
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=factorised_v3_%j.out
#SBATCH --error=factorised_v3_%j.out
#
# CPU only: the factorised 2->4 head's data check on catalog_v3 (the level item, moved to catalog_v3 by the user's call
# of 2026-10-25): every 2->4 class built as a 2->3 class plus a gluon, photon, Z or H (b beams through the beam-recoil map, a missing parent standalone compiled, the skipped listed), each
# against its parent's exact |M|^2 at the mapped events (tools/factorised_2to4.py --v3); $SCRATCH/zz_factorised_v3_all/ (arg 5: the 40 2->5 probes onto their 2->4 parents, zz_factorised_v3_2to5/)
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
NF="${1:-4}"; OUTD="$SCRATCH/zz_factorised_v3_all"; [ "$NF" = 5 ] && OUTD="$SCRATCH/zz_factorised_v3_2to5"
python tools/factorised_2to4.py --v3 --nfinal "$NF" --n 2000 --skip-done --out "$OUTD"
