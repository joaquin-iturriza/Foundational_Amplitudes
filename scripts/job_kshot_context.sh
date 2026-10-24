#!/bin/bash
#SBATCH --job-name=kshot_context
#SBATCH --cpus-per-task=5
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=kshot_context_%j.out
#SBATCH --error=kshot_context_%j.out
#
# D105 (the Level item's design (i)): the K-shot fits of the context vector on a catalog run trained with
# model.context_dim (scripts/job_catalog_norm.sh DlessInOrdQc short): per held-out process, c alone fitted on K = 1, 4,
# 16 of its validation events with everything else frozen, scored on the others (tools/kshot_context.py), against the
# plain K-event offset on the same draws; the shrunk offset of fig:kshot_level from the saved zero-shot q0 offline.
# Sets: the 43 new (the catalog's 139 held-out without a relabel in training, analysis/catalog_daniel/heldout_twins.json)
# and the 2->4 +g / +a probes (recipes/catalog_2to4_heldout_orders.yaml). Output: $SCRATCH/zz_kshot_context_<arm>/<set>/.
#   site submit ccin2p3 FA scripts/job_kshot_context.sh --hdr=--mem=32G -- [arm, default DlessInOrdQc] [smoke]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
ARM="${1:-DlessInOrdQc}"; SMOKE="${2:-}"
RUN="$PROJECT_DIR/runs/catalog_norm_short/$ARM"
PAR="$RUN"; ls "$RUN"/sigma_stage/models/model_run0_best.pt* >/dev/null 2>&1 && PAR="$RUN/sigma_stage"
ls "$PAR"/models/model_run0_best.pt* >/dev/null 2>&1 || { echo "no best checkpoint in $PAR"; exit 1; }
test -e "$PAR/context_names.json" || { echo "$PAR has no context_names.json (not a model.context_dim run)"; exit 1; }
OUT="$SCRATCH/zz_kshot_context_$ARM"; [ "$SMOKE" = smoke ] && OUT="$SCRATCH/zz_kshot_context_smoke_${ARM}_$SLURM_JOB_ID"
R="$SCRATCH/zz_kshot_context_recipes"; mkdir -p "$R/new43" "$R/2to4"
python - "$R" <<'PY'
import json, sys, yaml
R = sys.argv[1]
tw = set(json.load(open("analysis/catalog_daniel/heldout_twins.json")))
n = 0
for p in yaml.safe_load(open("recipes/catalog_daniel_heldout.yaml"))["processes"]:
    if p["name"] not in tw:
        yaml.safe_dump({"processes": [p]}, open(f"{R}/new43/{p['name']}.yaml", "w")); n += 1
m = 0
for p in yaml.safe_load(open("recipes/catalog_2to4_heldout_orders.yaml"))["processes"]:
    if p["name"].endswith(("_g", "_a")):
        yaml.safe_dump({"processes": [p]}, open(f"{R}/2to4/{p['name']}.yaml", "w")); m += 1
print(n, "new held-out,", m, "2->4 +g/+a recipes")
PY
for S in new43 2to4; do
    P=("$R/$S"/*.yaml); [ "$SMOKE" = smoke ] && P=("${P[@]:0:2}")
    mkdir -p "$OUT/$S"
    if [ "$SMOKE" = smoke ]; then
        python tools/kshot_context.py --parent "$PAR" --ks 1 4 --draws 2 --steps 20 --save-dir "$OUT/$S" "${P[@]}"
    else
        python tools/kshot_context.py --parent "$PAR" --save-dir "$OUT/$S" "${P[@]}"
    fi
done
