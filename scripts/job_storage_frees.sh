#!/bin/bash
#SBATCH --job-name=storage_frees
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=storage_frees_%j.out
#SBATCH --error=storage_frees_%j.out
#
# CPU only (cc_disk): the frees that lose nothing or are already decided, each with df before and after.
#   1. gzip the lhe_queue deletion manifests (Q/gc/*, read only by tests) older than 6 h, in place: lossless
#   2. D208: lhe_parts gc --parts on every queue (deletes only the parts of assembled processes; manifest under Q/gc/)
#   3. D136: free_space.py's events dirs (MadGraph *_events), manifest first in $SCRATCH/zz_cc_disk/
#   4. D550: model files of runs/zz_smoke runs older than 3 days and named nowhere (tools/delete_debris.py smoke_models)
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
O="$SCRATCH/zz_cc_disk"; mkdir -p "$O"
df_used() { echo "DF $1 $(df -B1M "$SCRATCH" | tail -1)"; }
df_used start
echo "GC_MANIFESTS before MiB $(du -sc --apparent-size --block-size=1M "$SCRATCH"/lhe_queue/*/gc | tail -1 | cut -f1)"
find "$SCRATCH"/lhe_queue/*/gc -type f ! -name "*.gz" -mmin +360 -print0 | xargs -0 -r gzip
echo "GC_MANIFESTS after MiB $(du -sc --apparent-size --block-size=1M "$SCRATCH"/lhe_queue/*/gc | tail -1 | cut -f1)"
df_used gzip
for q in "$SCRATCH"/lhe_queue/*/; do
    [ -d "$q/done" ] || { echo "GC_PARTS skip $(basename "$q"): not a queue"; continue; }
    echo "GC_PARTS $(basename "$q") $(python tools/lhe_parts.py gc --queue "$q" --parts 2>&1 | tail -1)"
done
df_used gc_parts
M="$O/free_manifest_$(date +%Y%m%d_%H%M).tsv"
python analysis/storage/free_space.py list > "$M" && python analysis/storage/free_space.py delete "$M" --kind events | grep -E "^#BATCH|^SKIP"
df_used events
python tools/delete_debris.py --classes smoke_models --manifest "$O/deleted_smoke_models.tsv"
df_used smoke_models
echo STORAGE_FREES_DONE
