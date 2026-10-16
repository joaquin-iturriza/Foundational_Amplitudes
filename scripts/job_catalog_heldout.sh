#!/bin/bash
#SBATCH --job-name=catalog_heldout
#SBATCH --cpus-per-task=5
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=catalog_heldout_%j.out
#SBATCH --error=catalog_heldout_%j.out
#
# The best pretraining of the arXiv:2606.23791 catalog sweep (D59, dancat_het_pre) on its held-out split (D45, "Their
# split, report both"): (1) the sigma-only stage (D70; tools/sigma_stage.py, the sweep predates it), (2) the zero-shot
# of the pretraining on each of the 139 held-out processes (recipes/catalog_daniel_heldout.yaml, one recipe per process
# written to $SCRATCH; tools/zero_shot_parent.py: one ZERO_SHOT line each). The parent standardizes per process, so a
# process it has no data of is scored up to an affine map (val_loss_affine = 1 - rho^2), not in log|M|^2 itself.
#   site submit ccin2p3 FA scripts/job_catalog_heldout.sh -- runs/dancat_het_pre/trial_0058
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
P="$1"
python tools/sigma_stage.py "$P" || echo "SIGMA_STAGE_FAILED $P"
R="$SCRATCH/zz_catalog_heldout_recipes"; mkdir -p "$R"
python - "$R" <<'PY'
import sys, yaml
r = yaml.safe_load(open("recipes/catalog_daniel_heldout.yaml"))
for p in r["processes"]:
    yaml.safe_dump({"processes": [p]}, open(f"{sys.argv[1]}/{p['name']}.yaml", "w"))
print(len(r["processes"]), "held-out recipes")
PY
python tools/zero_shot_parent.py --parent "$P" "$R"/dc_*.yaml
