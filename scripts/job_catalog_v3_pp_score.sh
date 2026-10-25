#!/bin/bash
#SBATCH --job-name=v3_pp_score
#SBATCH --cpus-per-task=5
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=v3_pp_score_%j.out
#SBATCH --error=v3_pp_score_%j.out
#
# The pp study (the user, 2026-10-25): a catalog_v3 short pretraining trained on the shaped windows
# (scripts/job_catalog_v3_short.sh, runs/catalog_v3_short/<arm>) read zero-shot on the 13 TeV MadEvent pools of the
# 1396 hadronic classes (recipes/catalog_v3_pp_test.yaml: 410 held-out, 986 training; each pp13_<class> is its class's
# |M|^2 function on pp-distributed events), 2000 val events each (tools/zero_shot_parent.py); predictions in
# $SCRATCH/zz_v3_pp_<arm>/. Args: <arm> [recipe, default the test set] [missing]
# Third arg "missing" (or MISSING_ONLY=1) scores only the recipe's probes with no readable prediction file yet (a read cut short, e.g. by a full disk).
# The job ends nonzero, without V3_PP_SCORE_DONE, when the scorer fails or any probe of the recipe has no
# readable prediction file afterwards (every such probe listed as V3_PP_MISSING); a ZERO_SHOT error line is a failed probe too.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
ARM="${1:?an arm of runs/catalog_v3_short}"; SPEC="${2:-recipes/catalog_v3_pp_test.yaml}"
[ "${3:-}" = missing ] && MISSING_ONLY=1
RUN="$PROJECT_DIR/runs/catalog_v3_short/$ARM"
ls "$RUN"/models/model_run0_best.pt* >/dev/null 2>&1 || { echo "no best checkpoint in $RUN"; exit 1; }
R="$SCRATCH/zz_v3_pp_recipes"; PRED="$SCRATCH/zz_v3_pp_$ARM"; mkdir -p "$R" "$PRED"
# one recipe per probe; prints the ones to score (all, or with MISSING_ONLY=1 those with no prediction file)
P=($(python - "$R" "$SPEC" "$PRED" "${MISSING_ONLY:-0}" <<'PY'
import os, sys, yaml
sys.path.insert(0, "scripts")
from pp_score_ok import ok
R, spec, pred, only = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
r = yaml.safe_load(open(spec))
top = {k: v for k, v in r.items() if k != "processes"}
todo = []
for p in r["processes"]:
    yaml.safe_dump({**top, "processes": [p]}, open(f"{R}/{p['name']}.yaml", "w"))
    if not (only and ok(f"{pred}/{p['name']}.npz")):
        todo.append(f"{R}/{p['name']}.yaml")
print(len(r["processes"]), "pp recipes,", len(todo), "to score", file=sys.stderr)
print(" ".join(todo))
PY
)) || { echo "V3_PP_SCORE_FAILED $ARM: recipe split"; exit 1; }
# the scorer's output kept on the node (not $SCRATCH, which may be the disk that filled), read for error lines
LOG=$(mktemp "${TMPDIR:-/tmp}/v3_pp_score_XXXXXX.log")
rc=0
if [ "${#P[@]}" -gt 0 ]; then
    python tools/zero_shot_parent.py --parent "$RUN" --save-dir "$PRED" "${P[@]}" 2>&1 | tee "$LOG"
    rc=${PIPESTATUS[0]}
fi
nerr=$(grep -c '^ZERO_SHOT {.*"error"' "$LOG" 2>/dev/null || true)
# every probe of the recipe must have its prediction file now, whatever was scored in this job
miss=$(python - "$SPEC" "$PRED" <<'PY'
import sys, yaml
sys.path.insert(0, "scripts")
from pp_score_ok import ok
for p in yaml.safe_load(open(sys.argv[1]))["processes"]:
    if not ok(f"{sys.argv[2]}/{p['name']}.npz"):
        print(p["name"])
PY
) || { echo "V3_PP_SCORE_FAILED $ARM: completeness check"; exit 1; }
for m in $miss; do echo "V3_PP_MISSING $ARM $m"; done
nmiss=$(printf '%s\n' $miss | grep -c . || true)
if [ "$rc" -ne 0 ] || [ "${nerr:-0}" -gt 0 ] || [ "$nmiss" -gt 0 ]; then
    echo "V3_PP_SCORE_FAILED $ARM: scorer exit $rc, ${nerr:-0} probe errors, $nmiss probes without a prediction"
    exit 1
fi
rm -f "$LOG"
echo "V3_PP_SCORE_DONE $ARM"
