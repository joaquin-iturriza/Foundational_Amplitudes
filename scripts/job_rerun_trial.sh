#!/bin/bash
#SBATCH --job-name=rerun_trial
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# Rerun a pruned trial from its own config.yaml (sweep/prune.py keeps every trial's config; CLAUDE.md: "Need
# another trial's model? Rerun it from its config"), cold, into <trial>_rerun next to it; then its Delta eval.
#     site submit <site> FA scripts/job_rerun_trial.sh -- runs/<sweep>/trial_N [<parent checkpoint>]
# The optional second argument re-points fine_tune.pretrained_path (a trial copied from another site names that
# site's parent path), relative to the checkout.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
T="$PROJECT_DIR/$1"
test -f "$T/config.yaml" || { echo "no $T/config.yaml"; exit 1; }
test -e "${T}_rerun" && { echo "${T}_rerun exists"; exit 1; }
PARENT=()
[ -n "$2" ] && PARENT=(fine_tune.pretrained_path="$PROJECT_DIR/$2")
python run.py --config-path "$T" --config-name config run_dir="${T}_rerun" warm_start_idx=null \
    training.is_dyhpo_run=false plot=true "${PARENT[@]}"
python tools/eval_delta.py --out-dir "$SCRATCH/zz_delta" "${T}_rerun"
