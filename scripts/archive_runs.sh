#!/bin/bash
#SBATCH --job-name=archive_runs
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Pack finished studies' run directories into one tar per study, to free inodes on a site with a file quota
# (Jean Zay $WORK: 500k files; the user approved archiving the finished studies, 2026-10-02). A family is
# runs/<family>_*, matched on the name's leading underscore-separated fields (runs/scaling_p1_* and
# runs/scaling_p1ext_* are separate families). For each family: tar into <dest>/<family>.tar, check that the
# archive lists exactly as many entries as the directories hold and that every member compares equal to the disk
# (tar --compare: content and metadata), and only then remove the directories. A family whose archive exists is
# skipped. Restore with:  tar -xf <dest>/<family>.tar -C "$PROJECT_DIR/runs" [member ...]
# (analysis/divergences/extract.sh reads two finetune_scaling trials: restore them first.)
# A finished sweep's checkpoint_index points into its runs: restore the family before any --extend of it.
#     site submit <site> FA scripts/archive_runs.sh -- <dest_dir> <family> [<family> ...]
#     site submit <site> FA scripts/archive_runs.sh -- --auto <dest_dir> <keep-prefix> [...]   (@STORE@ in dest = $STORE)
set -uo pipefail
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
# --auto <dest> <keep>...: every family in runs/ except those whose name starts with a <keep> prefix or equals a
# <keep> name (a study still in use, a fine-tune parent); the family of runs/a_b_c is a_b (its last field dropped),
# a directory without "_" is listed and left alone. Prints the families first, so the job log says what it took.
AUTO=0
if [ "${1:-}" = "--auto" ]; then AUTO=1; shift; fi
DEST=$(realpath -m "${1//@STORE@/$STORE}"); shift
mkdir -p "$DEST"
cd "$PROJECT_DIR/runs" || exit 1
if [ $AUTO = 1 ]; then
    keep=("$@")
    fams=()
    for d in */; do
        d=${d%/}
        skip=0
        for k in "${keep[@]}"; do [[ "$d" == "$k"* ]] && skip=1; done
        [ $skip = 1 ] && continue
        if [[ "$d" != *_* ]]; then echo "left alone (no family): $d"; continue; fi
        fams+=("${d%_*}")
    done
    set -- $(printf '%s\n' "${fams[@]}" | sort -u)
    echo "families to archive ($#): $*"
fi
for fam in "$@"; do
    dirs=$(ls -d "${fam}"_*/ 2>/dev/null | sed 's#/$##' | awk -F_ -v f="$fam" \
        '{n = split(f, a, "_"); k = $1; for (i = 2; i <= n; i++) k = k "_" $i; if (k == f) print}')
    if [ $AUTO = 1 ]; then   # a kept directory never goes into another family's archive
        dirs=$(for d in $dirs; do s=0; for k in "${keep[@]}"; do [[ "$d" == "$k"* ]] && s=1; done; [ $s = 0 ] && echo "$d"; done)
    fi
    if [ -z "$dirs" ]; then echo "$fam: no directories"; continue; fi
    out="$DEST/$fam.tar"
    if [ -e "$out" ]; then echo "$fam: $out exists, skipped"; continue; fi
    n_dirs=$(echo "$dirs" | wc -l)
    n_files=$(find $dirs | wc -l)
    if ! tar -cf "$out.part" $dirs; then echo "$fam: tar failed"; rm -f "$out.part"; continue; fi
    n_tar=$(tar -tf "$out.part" | wc -l)
    if [ "$n_tar" != "$n_files" ]; then
        echo "$fam: archive lists $n_tar entries, the directories hold $n_files: kept both"; continue
    fi
    if ! tar -df "$out.part" > "$DEST/$fam.compare.log" 2>&1; then
        echo "$fam: archive differs from the disk ($DEST/$fam.compare.log): kept both"; continue
    fi
    rm -f "$DEST/$fam.compare.log"
    mv "$out.part" "$out"
    rm -rf $dirs
    echo "$fam: $n_dirs directories, $n_files files -> $out ($(du -h "$out" | cut -f1)), originals removed"
done
