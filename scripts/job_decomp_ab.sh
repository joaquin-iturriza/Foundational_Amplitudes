#!/bin/bash
#SBATCH --job-name=decomp_ab
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# Decomposed targets (D48) at an MSE baseline trial's own HPs (CLAUDE.md A/B step 3), one seed per job; run dir
# runs/zz_decomp_ab/<sweep>__<trial>_<TAG>_s<seed>.
#     site submit <site> FA scripts/job_decomp_ab.sh -- runs/<sweep>/trial_N <seed> <TAG> [hydra overrides...]
# TAG names the arm: colour | colour_w0 | diagram set data.decomp_basis (and training.decomp_weight=0 for _w0);
# base reruns the baseline's own config on the current code (no decomposition). The baseline's target levers are
# switched off for every arm, the sign head too (a positive pool: it carries no loss there); for uubar_gg neither
# changes the target (no Z/t/H propagator, prepd_std asserted equal in the write-up). increment_steps is cleared
# (a DyHPO trial's fidelity sets the cosine horizon). Sidecars: tools/decomp_targets.py --recipe, ${DATA_DIR}/decomp.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
T="$PROJECT_DIR/$1"; S="$2"; TAG="$3"
OUT="$PROJECT_DIR/runs/zz_decomp_ab/$(basename $(dirname $T))__$(basename $T)_${TAG}_s$S"
test -e "$OUT" && { echo "$OUT exists"; exit 1; }
case "$TAG" in
  base)      ARM=() ;;
  colour)    ARM=(++data.decomp_basis=colour ++training.decomp_weight=1.0) ;;
  colour_w0) ARM=(++data.decomp_basis=colour ++training.decomp_weight=0.0) ;;
  diagram)   ARM=(++data.decomp_basis=diagram ++training.decomp_weight=1.0) ;;
  *) echo "unknown arm $TAG"; exit 1 ;;
esac
python run.py --config-path "$T" --config-name config run_dir="$OUT" warm_start_idx=null \
    training.is_dyhpo_run=false training.increment_steps=null training.result_path="$OUT/result.json" seed=$S plot=true \
    data.target_propagators=false training.sign_head=false "${ARM[@]}" "${@:4}"
