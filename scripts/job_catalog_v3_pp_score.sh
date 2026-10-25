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
# $SCRATCH/zz_v3_pp_<arm>/. Args: <arm> [recipe, default the test set]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
ARM="${1:?an arm of runs/catalog_v3_short}"; SPEC="${2:-recipes/catalog_v3_pp_test.yaml}"
RUN="$PROJECT_DIR/runs/catalog_v3_short/$ARM"
ls "$RUN"/models/model_run0_best.pt* >/dev/null 2>&1 || { echo "no best checkpoint in $RUN"; exit 1; }
R="$SCRATCH/zz_v3_pp_recipes"; PRED="$SCRATCH/zz_v3_pp_$ARM"; mkdir -p "$R" "$PRED"
python - "$R" "$SPEC" <<'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[2]))
top = {k: v for k, v in r.items() if k != "processes"}
for p in r["processes"]:
    yaml.safe_dump({**top, "processes": [p]}, open(f"{sys.argv[1]}/{p['name']}.yaml", "w"))
print(len(r["processes"]), "pp recipes")
PY
P=($(python -c "import sys, yaml; print(' '.join(f\"{sys.argv[1]}/{p['name']}.yaml\" for p in yaml.safe_load(open(sys.argv[2]))['processes']))" "$R" "$SPEC"))
python tools/zero_shot_parent.py --parent "$RUN" --save-dir "$PRED" "${P[@]}"
echo "V3_PP_SCORE_DONE $ARM"
