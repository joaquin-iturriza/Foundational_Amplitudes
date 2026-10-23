#!/bin/bash
# The supervisor's storage upkeep for one site (projects.toml `maintain`): sweep/prune.py there, at most every 6 h,
# its per-sweep manifests appended on the laptop (~/.local/share/ccorch/artifacts/FA/prune/<site>.jsonl).
set -o pipefail
site=$1
out=~/.local/share/ccorch/artifacts/FA/prune; mkdir -p "$out"
tmp=$(mktemp)
timeout 3000 site --timeout 2990 run --item '*' "$site" FA -- python sweep/prune.py --apply --submit-logs --caches --every 6 --exclude tp3_wadd_ tp3_waddm_ > "$tmp" 2>&1
rc=$?
# sweeps copied from another site have no result files here, so prune.py keeps all their models: the same rule from the
# trials' logs (tools/prune_copies.py; the user, 2026-10-23, "aren't we supposed to be pruning ... the models that
# don't win in a sweep?")
if [ $rc -eq 0 ] && [ "$site" = ccin2p3 ]; then
    timeout 1800 site --timeout 1790 run --item '*' "$site" FA -- python tools/prune_copies.py runs_from_lxplus --apply \
        | grep -E '^PRUNED_COPIES|^manifest' | sed 's/^/SUMMARY copies: /'
fi
grep '^PRUNE ' "$tmp" | sed 's/^PRUNE //' >> "$out/$site.jsonl"
grep '^SUMMARY' "$tmp"
[ $rc -eq 0 ] || tail -5 "$tmp"
rm -f "$tmp"
exit $rc
