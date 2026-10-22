#!/bin/bash
#SBATCH --job-name=catalog_heldout_preds
#SBATCH --cpus-per-task=5
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=catalog_heldout_preds_%j.out
#SBATCH --error=catalog_heldout_preds_%j.out
#
# The catalog pretraining's best trial on the 139 held-out processes again (as scripts/job_catalog_heldout.sh, no sigma
# stage), now keeping each event's prediction (tools/zero_shot_parent.py --save-dir) and scoring the 96 relabels in
# log|M|^2 with their training twin's statistics (--twins analysis/catalog_daniel/heldout_twins.json): the Delta plots
# and the per-process MSE figure, raw and after the best affine map (the user, 2026-10-22). Predictions in
# $SCRATCH/zz_catalog_heldout_preds/.
#   site submit ccin2p3 FA scripts/job_catalog_heldout_preds.sh -- runs/dancat_het_pre/trial_0058 [out dir name]
# The sigma head's calibration on the held-out: the parent's sigma_stage dir as the parent (mu frozen there, D70), into
# its own out dir (... -- runs/dancat_het_pre/trial_0058/sigma_stage zz_catalog_heldout_preds_sigstage).
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
P="$1"; OUTN="${2:-zz_catalog_heldout_preds}"
R="$SCRATCH/zz_catalog_heldout_recipes"; mkdir -p "$R"
python - "$R" <<'PY'
import sys, yaml
r = yaml.safe_load(open("recipes/catalog_daniel_heldout.yaml"))
for p in r["processes"]:
    yaml.safe_dump({"processes": [p]}, open(f"{sys.argv[1]}/{p['name']}.yaml", "w"))
print(len(r["processes"]), "held-out recipes")
PY
python tools/zero_shot_parent.py --parent "$P" --twins analysis/catalog_daniel/heldout_twins.json \
    --save-dir "$SCRATCH/$OUTN" "$R"/dc_*.yaml
