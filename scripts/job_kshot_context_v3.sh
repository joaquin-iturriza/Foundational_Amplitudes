#!/bin/bash
#SBATCH --job-name=kshot_v3
#SBATCH --cpus-per-task=5
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=kshot_v3_%j.out
#SBATCH --error=kshot_v3_%j.out
#
# The context vector's K-shot fits on catalog_v3 (the level item, moved to catalog_v3 by the user's call of 2026-10-25):
# on the catalog_v3 short run with the context vector (scripts/job_catalog_v3_short.sh dimensionless_context), per
# held-out process of one axis, c alone fitted on K = 1, 4, 16 of its validation events, everything else frozen, scored
# on the others, against the plain K-event offset on the same draws (tools/kshot_context.py, as on Daniel's catalog).
# Output: $SCRATCH/zz_kshot_v3/<axis>/. Args: <axis> (one of catalog_v3_heldout_<axis>.yaml) [k n]: part k of n of the
# axis's processes (every n-th from k, sorted by name), so a large axis runs as n parallel jobs
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
AX="${1:?a held-out axis}"; K="${2:-0}"; NP="${3:-1}"
PAR="$PROJECT_DIR/runs/catalog_v3_short/dimensionless_context"
ls "$PAR"/models/model_run0_best.pt* >/dev/null 2>&1 || { echo "no best checkpoint in $PAR"; exit 1; }
test -e "$PAR/context_names.json" || { echo "$PAR has no context_names.json"; exit 1; }
R="$SCRATCH/zz_kshot_v3_recipes/$AX"; OUT="$SCRATCH/zz_kshot_v3/$AX"; mkdir -p "$R" "$OUT"
python - "$R" "recipes/catalog_v3_heldout_$AX.yaml" <<'PY'
import os, sys, yaml
r = yaml.safe_load(open(sys.argv[2]))
top = {k: v for k, v in r.items() if k != "processes"}
for p in r["processes"]:
    f = f"{sys.argv[1]}/{p['name']}.yaml"           # through a temp file: the axis's parts write them at once
    yaml.safe_dump({**top, "processes": [p]}, open(f + ".tmp%d" % os.getpid(), "w"))
    os.replace(f + ".tmp%d" % os.getpid(), f)
print(len(r["processes"]), "recipes")
PY
ALL=("$R"/*.yaml); P=()
for i in "${!ALL[@]}"; do [ $((i % NP)) = "$K" ] && P+=("${ALL[$i]}"); done
echo "part $K of $NP: ${#P[@]} of ${#ALL[@]} processes"
python tools/kshot_context.py --parent "$PAR" --save-dir "$OUT" "${P[@]}"
echo "KSHOT_V3_DONE $AX $K/$NP"
