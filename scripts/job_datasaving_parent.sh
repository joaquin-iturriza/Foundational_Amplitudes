#!/bin/bash
#SBATCH --job-name=dsparent
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# pretrain_data_saving: a parent of the stage-2 fine-tunes (D393): every pool of
# recipes/catalog_v4_datasaving_parent_<which>.yaml on its first 1e4 train events, 5 epochs at the parent's batch
# 16384, cosine at HPO rule 2's lr centre 1e-2 (T/3000)^-0.55 (see lr=<value> below):
#   heldout  1422 pools (recipes/gen_datasaving_parent.py, D400): trial 1262's training recipe plus catalog_v4's
#            physical-mass top classes, the e+e- > t t~ family and e+e- > u u~'s class out (the LO fine-tunes' parent):
#            T = ceil(5 x 1422e4 / 16384) = 4340, lr 1e-2 (T/3000)^-0.55 = 8.162e-3
#   treesin  1424 pools, the held-out recipe plus e+e- > u u~ and t t~ LO (the one-loop fine-tunes' parent: "the one
#            loops with the tree levels in the pretraining"): T = 4346, lr 8.156e-3
# Base config:
# trial 1262 of cov_v4_long_hpo8_cc_68224 (model, HPs, HETEROSC + sigma stage); in memory (data.memmap=false: the
# 1e4-per-pool train split is ~1.42e7 events, 10 GB host memory at 1.13e7 in r4, and no memmap cache exists for this recipe).
#     site submit <site|auto> FA scripts/job_datasaving_parent.sh -- <heldout|treesin> [key=value ...] [tag=<name>]
# The input flags of D392 (kit_benchmark's arms with a gain) go as arguments with ++ (trial 1262's config predates
# the keys, so a plain override fails "not in struct"), e.g. ++model.input_soft_eps=1e-8
# ++model.input_pair_logs=6 (6 = the recipe's largest multiplicity, catalog_v4 goes to 2->4, wrappers.py asserts it
# at 6; beams first in every pool; the two LO targets add 4-particle pools). Run dir
# runs/data_saving/dsave_parent_<which>[_<tag>]. lr=<value> replaces the default lr, HPO rule 2's centre 1e-2 (T/3000)^-0.55
# (at the law 7.82114e-3 (T/17056)^-0.55, 1.88e-2, both r3 parents diverged near step 900-1000, 2026-11-11). iters=<n>
# replaces T (a smoke run, D395).
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
WHICH="$1"; shift
case "$WHICH" in heldout) T=4340; LR=8.162e-3 ;; treesin) T=4346; LR=8.156e-3 ;; *) echo "heldout or treesin"; exit 2 ;; esac
PARENT="runs/catalog_v4_long/best_cov_v4_long_hpo8_cc_68224_trial_1262"
RECIPE="$PROJECT_DIR/recipes/catalog_v4_datasaving_parent_${WHICH}.yaml"
[ -e "$RECIPE" ] && [ -e "$PROJECT_DIR/$PARENT/config.yaml" ] || { echo "no $RECIPE or $PARENT/config.yaml"; exit 2; }
TAG=; ARGS=(); for x in "$@"; do case "$x" in tag=*) TAG="_${x#tag=}";; lr=*) LR="${x#lr=}";; iters=*) T="${x#iters=}";; *) ARGS+=("$x");; esac; done
OUT="$PROJECT_DIR/runs/data_saving/dsave_parent_${WHICH}${TAG}"
echo "DATASAVING_PARENT recipe=$RECIPE base=$PARENT args: ${ARGS[*]}"
python run.py --config-path "$PROJECT_DIR/$PARENT" --config-name config run_dir="$OUT" run_name=null \
    data.processes_file="$RECIPE" data.train_subsample=10000 data.memmap=false fine_tune.pretrained_path=null \
    warm_start_idx=null training.iterations="$T" training.lr="$LR" training.is_dyhpo_run=false "${ARGS[@]}"
RC=$?
echo "DATASAVING_PARENT_DONE rc=$RC $OUT"
[ "$RC" -eq 0 ] && echo "RUN_DONE $OUT"
exit $RC
