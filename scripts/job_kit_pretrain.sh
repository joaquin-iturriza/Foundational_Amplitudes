#!/bin/bash
#SBATCH --job-name=kitpre
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# kit_benchmark: a catalog_v4 pretraining with the KIT targets' families held out, at the parent pretraining's
# setup (trial 1262 of cov_v4_long_hpo8_cc_68224: its saved config.yaml is the base, so model, HPs, steps, loss and
# sigma stage are the parent's); only the recipe changes (recipes/gen_kit_pretrain.py).
#     site submit <site|auto> FA scripts/job_kit_pretrain.sh -- <heldout|fair_nlo> [key=value ...]
# heldout   recipes/catalog_v4_kit_pretrain_heldout.yaml: protocol A's pretrained parent (no ee->tt, ee->qqbar g, ee->qqbar gg)
# fair_nlo  recipes/catalog_v4_kit_pretrain_fair_nlo.yaml: Bakar's fair NLO parent (every tree, the target's tree ee->tt
#           included, 10 one-loop pools)
# Run dir runs/kit/kit_pretrain_<recipe>[_<tag>] (tag=<name> argument).
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
WHICH="$1"; shift
case "$WHICH" in heldout|fair_nlo) ;; *) echo "heldout or fair_nlo"; exit 2 ;; esac
PARENT="runs/catalog_v4_long/best_cov_v4_long_hpo8_cc_68224_trial_1262"
RECIPE="$PROJECT_DIR/recipes/catalog_v4_kit_pretrain_${WHICH}.yaml"
[ -e "$RECIPE" ] && [ -e "$PROJECT_DIR/$PARENT/config.yaml" ] || { echo "no $RECIPE or $PARENT/config.yaml"; exit 2; }
TAG=; ARGS=(); for x in "$@"; do case "$x" in tag=*) TAG="_${x#tag=}";; *) ARGS+=("$x");; esac; done
OUT="$PROJECT_DIR/runs/kit/kit_pretrain_${WHICH}${TAG}"
echo "KIT_PRETRAIN recipe=$RECIPE parent=$PARENT args: ${ARGS[*]}"
python run.py --config-path "$PROJECT_DIR/$PARENT" --config-name config run_dir="$OUT" run_name=null \
    data.processes_file="$RECIPE" fine_tune.pretrained_path=null warm_start_idx=null training.iterations=68224 training.is_dyhpo_run=false "${ARGS[@]}"
RC=$?
echo "KIT_PRETRAIN_DONE rc=$RC $OUT"
# the supervisor's clean-end marker (~/.config/ccorch/projects.toml end_markers), only on success
[ "$RC" -eq 0 ] && echo "RUN_DONE $OUT"
exit $RC
