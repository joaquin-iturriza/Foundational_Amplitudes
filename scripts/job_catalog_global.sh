#!/bin/bash
#SBATCH --job-name=catalog_global
#SBATCH --cpus-per-task=5
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=catalog_global_%j.out
#SBATCH --error=catalog_global_%j.out
#
# The catalog pretraining with GLOBAL target preprocessing, so a held-out process is scored zero-shot on its own
# (the user, 2026-10-05: "at some point a model with global preprocessing has to be trained so we can do some zero
# shot tests"; 2026-10-22: "having a global preprocessing was the way"). The best trial's own config
# (runs/dancat_het_pre/trial_0058, hp58's HPs: CLAUDE.md A/B step 3) with one (mu, sigma) pooled over the 1352
# processes and the mean affine in the particle count (data.shared_standardization + data.affine_multiplicity_std, as
# the rung-9 zero-shot parents that placed the level best, fig:zero_shot_parts). Then the 139 held-out processes:
# tools/zero_shot_parent.py reads the prediction with the parent's mu_N, sigma (no fit), per-event predictions in
# $SCRATCH/zz_catalog_global_preds/.
#   site submit ccin2p3 FA scripts/job_catalog_global.sh --hdr=--mem=40G
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
SRC=runs/dancat_het_pre/trial_0058
OUT="$PROJECT_DIR/runs/catalog_global/hp58_affine_std"
if [ ! -e "$OUT/sigma_stage/models" ]; then
    test -e "$OUT" && mv "$OUT" "$OUT.partial_$SLURM_JOB_ID"
    python run.py --config-path "$PROJECT_DIR/$SRC" --config-name config run_dir="$OUT" run_name=null warm_start_idx=null \
        base_dir="$PROJECT_DIR" training.is_dyhpo_run=false training.increment_steps=null \
        training.result_path="$OUT/result.json" plot=true \
        ++data.shared_standardization=true ++data.affine_multiplicity_std=true || exit 1
fi
R="$SCRATCH/zz_catalog_heldout_recipes"; mkdir -p "$R"
python - "$R" <<'PY'
import sys, yaml
r = yaml.safe_load(open("recipes/catalog_daniel_heldout.yaml"))
for p in r["processes"]:
    yaml.safe_dump({"processes": [p]}, open(f"{sys.argv[1]}/{p['name']}.yaml", "w"))
print(len(r["processes"]), "held-out recipes")
PY
python tools/zero_shot_parent.py --parent "$OUT" --save-dir "$SCRATCH/zz_catalog_global_preds" "$R"/dc_*.yaml
