#!/bin/bash
#SBATCH --job-name=rerun_trial
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# Rerun a pruned trial from its own config.yaml (sweep/prune.py keeps every trial's config; CLAUDE.md: "Need
# another trial's model? Rerun it from its config"), cold, into <trial>_rerun next to it; then its Delta eval.
#     site submit <site> FA scripts/job_rerun_trial.sh -- runs/<sweep>/trial_N [<parent checkpoint>|-] [<n>]
# The optional third argument n reruns it again into <trial>_rerun<n> (a repeat of a rerun off its record).
# The optional second argument re-points fine_tune.pretrained_path (a trial copied from another site names that
# site's parent path), relative to the checkout.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
T="$PROJECT_DIR/$1"
test -f "$T/config.yaml" || { echo "no $T/config.yaml"; exit 1; }
R="${T}_rerun${3:-}"
test -e "$R" && { echo "$R exists"; exit 1; }
PARENT=()
# a copied trial names the other site's recipe path: re-root it on this checkout
PF=$(grep -m1 "processes_file:" "$T/config.yaml" | sed "s/.*processes_file: *//")
[ -n "$PF" ] && [ ! -e "$PF" ] && PARENT+=(data.processes_file="$PROJECT_DIR/recipes/${PF##*/recipes/}")
[ -n "$2" ] && [ "$2" != "-" ] && PARENT+=(fine_tune.pretrained_path="$PROJECT_DIR/$2")
# a fine-tune from before the head map (2026-11-12) loaded the parent's readout unchanged: rerun it the same way
grep -q "pretrained_path:" "$T/config.yaml" && ! grep -q "map_head:" "$T/config.yaml" && PARENT+=(+fine_tune.map_head=false)
# Hydra's own files (.hydra/, run.log) on the node's local disk: EOS refused new files there twice (I/O error 5,
# FA-lxplus-9f778f21, 4b765eb0, a18ab984) before the run started
python run.py --config-path "$T" --config-name config run_dir="$R" warm_start_idx=null \
    hydra.run.dir="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}/hydra_rerun_$$" \
    training.is_dyhpo_run=false plot=true training.result_path="$R/result.json" "${PARENT[@]}"
OUT=$(python tools/eval_delta.py --out-dir "$SCRATCH/zz_delta" "$R"); echo "$OUT"
# the supervisor's end marker: RUN_SUMMARY carries no rc field, so a clean rerun says so here
grep -q "^EVAL_DELTA_DONE 1/1" <<<"$OUT" && echo RUN_DONE
