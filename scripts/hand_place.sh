#!/bin/bash
# Record a hand placement for one site in the placement guard's plan file, when the user has decided where a batch
# runs and `site pick` cannot plan it (e.g. Jean Zay's probe timing out while the site works). The user's words go into
# the plan; placement_guard.sh then allows N named submissions there for its usual 60 min.
#     bash scripts/hand_place.sh <project> <site> <n_jobs> "<the user's words>"
set -e
P="$1"; SITE="$2"; N="$3"; WHY="$4"
[ -n "$P" ] && [ -n "$SITE" ] && [ -n "$N" ] && [ -n "$WHY" ] || { echo "usage: $0 <project> <site> <n> \"<user's words>\""; exit 2; }
F="$HOME/.local/share/ccorch/placement/$P.json"
python3 - "$F" "$SITE" "$N" "$WHY" <<'PY'
import json, os, sys, time
f, site, n, why = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
d = json.load(open(f)) if os.path.exists(f) else {"project": os.path.basename(f)[:-5]}
d.update(ts=int(time.time()), site=site, quota={site: n}, hand=True, note=why)
json.dump(d, open(f, "w"))
print(f"hand placement: {site} x{n} ({why})")
PY
