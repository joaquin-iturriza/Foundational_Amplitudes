#!/bin/bash
#SBATCH --job-name=delete_debris
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=delete_debris_%j.out
#SBATCH --error=delete_debris_%j.out
#
# CPU only: deletes our own debris under D550 (tools/delete_debris.py: old smokes and probes, old-format memmap keys no
# live job names, scratch run dirs older than 3 days; never the scheduler logs, which analyses parse). Manifest first, in $SCRATCH/zz_cc_disk/deleted.tsv. DEBRIS_ARGS="--dry-run" lists only.

_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
mkdir -p "$SCRATCH/zz_cc_disk"
df -B1M /sps/lpnhe | tail -1
python tools/delete_debris.py --manifest "$SCRATCH/zz_cc_disk/deleted.tsv" ${DEBRIS_ARGS:-}
df -B1M /sps/lpnhe | tail -1
