#!/bin/bash
#SBATCH --job-name=het_ab
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# The user's heteroscedastic loss (losses.heteroscedastic_loss: plain Gaussian NLL, heterosc_beta=0, selection on the
# NLL; sign_head off: these probes are positive, where it carries no loss) at an MSE baseline trial's own HPs (CLAUDE.md A/B step 3), one seed per job; run dir runs/zz_het_ab/<sweep>__<trial>_s<seed>.
#     site submit <site> FA scripts/job_het_ab.sh -- runs/<sweep>/trial_N <seed> [TAG [hydra overrides...]]
# TAG (optional) is appended to the run dir, e.g. t16k with training.iterations=16000 (decision D61: the GP sweep's best
# trial at 16k and 32k, no sweep). increment_steps is cleared: a DyHPO trial's config carries its fidelity there, and it
# sets the cosine horizon, so a longer run would otherwise restart the cosine after the trial's steps.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
T="$PROJECT_DIR/$1"; S="$2"; TAG="${3:+_$3}"
OUT="$PROJECT_DIR/runs/zz_het_ab/$(basename $(dirname $T))__$(basename $T)_s$S$TAG"
test -e "$OUT" && { echo "$OUT exists"; exit 1; }
python run.py --config-path "$T" --config-name config run_dir="$OUT" warm_start_idx=null \
    training.is_dyhpo_run=false training.increment_steps=null training.result_path="$OUT/result.json" seed=$S plot=true \
    training.loss=HETEROSC training.heterosc_beta=0.0 ++training.heterosc_select=nll \
    ++model.net.sigma_after_pool=true ++model.net.detach_sigma_backbone=false \
    training.sign_head=false evaluation.save_predictions=true ++training.track_train_eval=true "${@:4}"
