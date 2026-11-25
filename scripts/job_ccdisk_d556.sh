#!/bin/bash
#SBATCH --job-name=ccdisk_d556
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=ccdisk_d556_%j.out
#SBATCH --error=ccdisk_d556_%j.out
#
# CPU only (cc_disk, D556):
#   1. delete the model files (*.pt, *.pt.gz, *.ckpt, *.pth) of the superseded fine-tune sets zz_catft, zz_catft_lowloss2,
#      zz_catft_lowloss3 (results files, logs, configs stay); manifest first in $SCRATCH/zz_cc_disk/deleted_catft.tsv
#   2. read who reads the caches (analysis/storage/cache_readers.py: per memmap key its readers and which in-use recipes
#      its train pools belong to) and census the training cache against the in-use recipes (cache_census.py): reads
#      only, nothing of the caches is deleted here
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
O="$SCRATCH/zz_cc_disk"; mkdir -p "$O"
echo "DF start $(df -B1M "$SCRATCH" | tail -1)"
M="$O/deleted_catft.tsv"
find "$SCRATCH/zz_catft" "$SCRATCH/zz_catft_lowloss2" "$SCRATCH/zz_catft_lowloss3" -type f \
    \( -name "*.pt" -o -name "*.pt.gz" -o -name "*.ckpt" -o -name "*.pth" \) -printf "%s\t%p\n" > "$M"
awk -F'\t' '{s+=$1;n++} END {printf "CATFT listed %d model files, %.2f GiB\n", n, s/2^30}' "$M"
cut -f2 "$M" | tr '\n' '\0' | xargs -0 -r rm -f --
echo "DF catft $(df -B1M "$SCRATCH" | tail -1)"
# in use: the recipes of the last 14 days' run and sweep configs, plus every recipe the not-done plan entries name
# (analysis/storage/open_recipes_2026-11-25.txt, from plan show of each entry, held ones included)
python analysis/storage/cache_readers.py --days 14 --in-use-file analysis/storage/open_recipes_2026-11-25.txt > "$O/cache_readers.txt"; cat "$O/cache_readers.txt"
INUSE="$(grep '^INUSE' "$O/cache_readers.txt" | cut -f2)"
set -f                     # the in-use globs go to cache_census as written
python analysis/storage/cache_census.py --in-use $INUSE --days 14 --workers 4 > "$O/census_d556.tsv" 2> "$O/census_d556.err"
tail -3 "$O/census_d556.err"
awk -F'\t' 'NR>1 {s[$8]+=$2; n[$8]++} END {for (k in s) printf "CENSUS %s %d files %.2f GiB\n", k, n[k], s[k]/2^30}' "$O/census_d556.tsv"
awk -F'\t' 'NR>1 && $8=="delete" {split($7,e,"/"); s[$6]+=$2} END {for (k in s) printf "CENSUS_DELETE_BY_RECIPE %.2f GiB\t%s\n", s[k]/2^30, k}' "$O/census_d556.tsv" | sort -rn | head -30
awk -F'\t' 'NR>1 && $8=="keep" {s[$9]+=$2} END {for (k in s) printf "CENSUS_KEEP_BY_REASON %.2f GiB\t%s\n", s[k]/2^30, k}' "$O/census_d556.tsv" | sort -rn | head -10
echo D556_READ_DONE
