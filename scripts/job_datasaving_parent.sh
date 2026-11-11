#!/bin/bash
#SBATCH --job-name=dsparent
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# pretrain_data_saving: the held-out parent of the stage-2 fine-tunes (D393): every pool of
# recipes/catalog_v4_datasaving_parent_heldout.yaml (1133 pools, the four Fig. 22 targets' equality classes out) on its
# first 1e4 train events, 5 epochs: ceil(5 x 1133 x 1e4 / 16384) = 3458 steps at the parent's batch 16384, cosine
# with the parent's lr law at that T, 7.82114e-3 (3458/17056)^-0.55 = 1.88e-2 (job_data_saving.sh's law). Base config:
# trial 1262 of cov_v4_long_hpo8_cc_68224 (model, HPs, HETEROSC + sigma stage); in memory (data.memmap=false: the
# 1e4-per-pool train split is ~1.1e7 events, and no memmap cache exists for this recipe).
#     site submit <site|auto> FA scripts/job_datasaving_parent.sh -- [key=value ...] [tag=<name>]
# The input flags of D392 (kit_benchmark's arms with a gain) go as arguments, e.g. model.input_soft_eps=1e-8
# model.input_pair_logs=6 (6 = the recipe's largest multiplicity, checked on CC 2026-11-11: 100 pools of 4, 319 of
# 5, 714 of 6 particles; beams first in every pool). Run dir runs/data_saving/dsave_parent_heldout[_<tag>].
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
PARENT="runs/catalog_v4_long/best_cov_v4_long_hpo8_cc_68224_trial_1262"
RECIPE="$PROJECT_DIR/recipes/catalog_v4_datasaving_parent_heldout.yaml"
[ -e "$RECIPE" ] && [ -e "$PROJECT_DIR/$PARENT/config.yaml" ] || { echo "no $RECIPE or $PARENT/config.yaml"; exit 2; }
TAG=; ARGS=(); for x in "$@"; do case "$x" in tag=*) TAG="_${x#tag=}";; *) ARGS+=("$x");; esac; done
OUT="$PROJECT_DIR/runs/data_saving/dsave_parent_heldout${TAG}"
echo "DATASAVING_PARENT recipe=$RECIPE base=$PARENT args: ${ARGS[*]}"
python run.py --config-path "$PROJECT_DIR/$PARENT" --config-name config run_dir="$OUT" run_name=null \
    data.processes_file="$RECIPE" data.train_subsample=10000 data.memmap=false fine_tune.pretrained_path=null \
    warm_start_idx=null training.iterations=3458 training.lr=1.88126e-2 training.is_dyhpo_run=false "${ARGS[@]}"
RC=$?
echo "DATASAVING_PARENT_DONE rc=$RC $OUT"
[ "$RC" -eq 0 ] && echo "RUN_DONE $OUT"
exit $RC
