#!/bin/bash
# Record a hand placement for one site in the placement guard's plan file, when the user has decided where a batch
# runs and `site pick` cannot plan it (e.g. Jean Zay's probe timing out while the site works). The placement must be a
# decision in force: an `approve` whose quote or covers names the site (`site decisions`); its words go into the plan,
# and placement_guard.sh then allows N named submissions there for its usual 60 min.
#     bash scripts/hand_place.sh <project> <site> <n_jobs> D<n>
set -e
P="$1"; SITE="$2"; N="$3"; DEC="$4"
[ -n "$P" ] && [ -n "$SITE" ] && [ -n "$N" ] && [[ "$DEC" =~ ^D[0-9]+$ ]] || { echo "usage: $0 <project> <site> <n> D<n>"; exit 2; }
LINE=$(site decisions --project "$P" 2>/dev/null | grep -E "^ +$DEC +$P +approve " || true)
[ -n "$LINE" ] || { echo "refused: $DEC is not an approve decision in force for $P"; exit 1; }
SITE_RE=$(echo "$SITE" | sed 's/jeanzay/jean[ _-]?zay/')
echo "$LINE" | grep -qiE "$SITE_RE" || { echo "refused: $DEC does not name $SITE"; exit 1; }
F="$HOME/.local/share/ccorch/placement/$P.json"
python3 - "$F" "$SITE" "$N" "$LINE" <<'PY'
import json, os, sys, time
f, site, n, why = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].strip()
d = json.load(open(f)) if os.path.exists(f) else {"project": os.path.basename(f)[:-5]}
d.update(ts=int(time.time()), site=site, quota={site: n}, hand=True, note=why)
json.dump(d, open(f, "w"))
print(f"hand placement: {site} x{n} ({why[:120]})")
PY
