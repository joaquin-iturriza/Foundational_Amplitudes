#!/bin/bash
# The supervisor's storage upkeep for one site (projects.toml `maintain`): sweep/prune.py there, at most every 6 h,
# its per-sweep manifests appended on the laptop (~/.local/share/ccorch/artifacts/FA/prune/<site>.jsonl).
set -o pipefail
site=$1
out=~/.local/share/ccorch/artifacts/FA/prune; mkdir -p "$out"
tmp=$(mktemp)
prune="python sweep/prune.py --apply --submit-logs --caches --every 6 --exclude tp3_wadd_ tp3_waddm_"
# sweeps copied from another site have no result files here, so prune.py keeps all their models: the same rule from the
# trials' logs (tools/prune_copies.py; the user, 2026-10-23, "aren't we supposed to be pruning ... the models that
# don't win in a sweep?")
copies='python tools/prune_copies.py runs_from_lxplus --apply | grep -E "^PRUNED_COPIES|^manifest" | sed "s/^/SUMMARY copies: /"'
if [ "$site" = ccin2p3 ]; then
    # CC's login nodes are not for scripts (its support, 2026-11-24, D544): both as one CPU job, each supervisor pass
    # reading the job the previous pass started (exit 75 = not ended yet, passed on: the supervisor skips it)
    timeout 300 site --timeout 4780 run --recurring --item '*' "$site" FA -- "$prune; rc=\$?; [ \$rc -eq 0 ] && { $copies; }; exit \$rc" > "$tmp" 2>&1
    rc=$?
    if [ $rc -eq 75 ] && grep -q CCJ_NO_RESULT "$tmp"; then cat "$tmp"; rm -f "$tmp"; exit 75; fi
else
    timeout 3000 site --timeout 2990 run --item '*' "$site" FA -- $prune > "$tmp" 2>&1
    rc=$?
fi
grep '^PRUNE ' "$tmp" | sed 's/^PRUNE //' >> "$out/$site.jsonl"
grep '^SUMMARY' "$tmp"
[ $rc -eq 0 ] || tail -5 "$tmp"
rm -f "$tmp"
exit $rc
