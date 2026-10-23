#!/bin/bash
#SBATCH --job-name=pack_archive
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --output=pack_archive_%j.out
#SBATCH --error=pack_archive_%j.out
# CPU only. Packs run directories into one tar per family under $SCRATCH/zz_archive_<tag>/ for a site-to-site copy to
# Jean Zay's STORE (D96, the user 2026-10-23: "archive and delete what you proposed before"; D9: old runs to STORE).
# Each tar is checked member by member against the disk (tar --compare) and its sha256 written next to it. Nothing is
# removed here: the directories go only after the copy's checksum is verified on the other side.
#     site submit ccin2p3 FA scripts/job_pack_archive.sh -- <tag> <list file of dirs relative to the dir above the checkout> <family>...
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
set -uo pipefail
TAG="$1"; LIST="$PROJECT_DIR/$2"; shift 2
BASE="$(dirname "$PROJECT_DIR")"
OUT="$SCRATCH/zz_archive_$TAG"; mkdir -p "$OUT"
rc=0
for fam in "$@"; do
    mapfile -t D < <(grep -E "/${fam}[^/]*$" "$LIST")
    [ ${#D[@]} -eq 0 ] && { echo "no dirs for $fam"; continue; }
    t="$OUT/$fam.tar"
    ( cd "$BASE" && tar -cf "$t" "${D[@]}" ) || { echo "tar failed: $fam"; rc=1; continue; }
    ( cd "$BASE" && tar --compare -f "$t" >/dev/null ) || { echo "compare failed: $fam"; rc=1; continue; }
    n=$(tar -tf "$t" | wc -l)
    sha256sum "$t" > "$t.sha256"
    echo "PACKED $fam: ${#D[@]} dirs, $n members, $(du -h "$t" | cut -f1), $(cut -c1-16 "$t.sha256")"
done
exit $rc
