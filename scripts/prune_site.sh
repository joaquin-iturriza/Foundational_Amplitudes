#!/bin/bash
# The supervisor's storage upkeep for one site (projects.toml `maintain`): sweep/prune.py there, at most every 6 h,
# its per-sweep manifests appended on the laptop (~/.local/share/ccorch/artifacts/FA/prune/<site>.jsonl).
set -o pipefail
site=$1
out=~/.local/share/ccorch/artifacts/FA/prune; mkdir -p "$out"
tmp=$(mktemp)
timeout 3000 site --timeout 2990 run --item '*' "$site" FA -- python sweep/prune.py --apply --submit-logs --caches --every 6 > "$tmp" 2>&1
rc=$?
grep '^PRUNE ' "$tmp" | sed 's/^PRUNE //' >> "$out/$site.jsonl"
grep '^SUMMARY' "$tmp"
[ $rc -eq 0 ] || tail -5 "$tmp"
rm -f "$tmp"
exit $rc
