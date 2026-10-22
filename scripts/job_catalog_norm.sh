#!/bin/bash
#SBATCH --job-name=catalog_norm
#SBATCH --cpus-per-task=5
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=catalog_norm_%j.out
#SBATCH --error=catalog_norm_%j.out
#
# D83 (the user: "send the A against B"): the catalog pretraining at hp58's configuration and HPs
# (runs/dancat_het_pre/trial_0058; CLAUDE.md A/B step 3) under a normalization that needs no data of a process, then
# the 139 held-out processes read with it (tools/zero_shot_parent.py, per-event predictions in $SCRATCH/zz_catalog_norm_<arm>/):
#   B  one standardization shared by all processes (data.shared_standardization)
#   A  the same after each event's |M|^2 is multiplied by (s0/s_hat)^(4-N)/C_p (data.target_physics_factor; s0 =
#      (17 GeV)^2, the reference fig:norm_factor chose on the training processes)
# 50 validations (D79). smoke: 19 catalog processes, 40 steps, 2 held-out processes (a test of the code path only).
#   site submit ccin2p3 FA scripts/job_catalog_norm.sh --hdr=--mem=40G -- A|B [smoke]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
ARM="${1:?A or B}"; SMOKE="${2:-}"
SRC=runs/dancat_het_pre/trial_0058
NORM=(++data.shared_standardization=true ++data.affine_multiplicity_std=false)
[ "$ARM" = A ] && NORM+=(++data.target_physics_factor=true ++data.target_physics_s0=289.0)
EXTRA=(training.validate_frac=0.02)
# the memory-mapped cache on the node's local disk: /sps/lpnhe was 98% full (2026-10-22, the first A run failed building
# it there); built by this job, removed when it exits
MMAP="${TMPDIR:-/tmp}/fa_mmap_${SLURM_JOB_ID}"; mkdir -p "$MMAP"; trap 'rm -rf "$MMAP"' EXIT
df -h "$MMAP" | tail -1
NORM+=(++data.memmap_dir="$MMAP")
OUT="$PROJECT_DIR/runs/catalog_norm/hp58_$ARM"; PRED="$SCRATCH/zz_catalog_norm_$ARM"
R="$SCRATCH/zz_catalog_heldout_recipes"; mkdir -p "$R"
if [ -n "$SMOKE" ]; then
    OUT="$SCRATCH/zz_catalog_norm_smoke_${ARM}_$SLURM_JOB_ID"; PRED="$OUT/preds"
    SUB="$OUT.recipe.yaml"
    python - "$SUB" <<'PY'
import random, sys, yaml
r = yaml.safe_load(open("recipes/catalog_daniel_train.yaml"))
random.Random(0).shuffle(r["processes"]); r["processes"] = r["processes"][:19]
yaml.safe_dump(r, open(sys.argv[1], "w"))
PY
    EXTRA=(training.iterations=40 training.validate_frac=0.5 evaluate=false data.processes_file="$SUB"
           ++training.heterosc_sigma_stage_steps=10)
fi
if [ ! -e "$OUT/sigma_stage/models" ]; then
    test -e "$OUT" && mv "$OUT" "$OUT.partial_$SLURM_JOB_ID"
    python run.py --config-path "$PROJECT_DIR/$SRC" --config-name config run_dir="$OUT" run_name=null warm_start_idx=null \
        base_dir="$PROJECT_DIR" training.is_dyhpo_run=false training.increment_steps=null \
        training.result_path="$OUT/result.json" plot=true "${NORM[@]}" "${EXTRA[@]}" || exit 1
fi
grep -a -m1 "target_physics_factor ON" "$OUT"/out_*.log 2>/dev/null
python - "$R" <<'PY'
import sys, yaml
r = yaml.safe_load(open("recipes/catalog_daniel_heldout.yaml"))
for p in r["processes"]:
    yaml.safe_dump({"processes": [p]}, open(f"{sys.argv[1]}/{p['name']}.yaml", "w"))
print(len(r["processes"]), "held-out recipes")
PY
P=("$R"/dc_*.yaml); [ -n "$SMOKE" ] && P=("$R/dc_scx_wmhh.yaml" "$R/dc_ssx_vevex.yaml")
python tools/zero_shot_parent.py --parent "$OUT" --save-dir "$PRED" "${P[@]}"
