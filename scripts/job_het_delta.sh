#!/bin/bash
#SBATCH --job-name=het_delta
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# D47 (the user: "the largest D cases for the delta plots I mentioned should be retrained with het loss, to also show
# pulls"; released by D71): one Delta-plot cell (analysis/transfer/delta_audit.json, paper_8k) retrained with the user's
# heteroscedastic loss at the cell's own config and HPs (CLAUDE.md A/B step 3), seed as the cell; a fine-tune cell starts
# from the same parent, its 1-channel readout grown to (mu, sigma) with mu kept (base_experiment). run.py ends it with
# the sigma-only stage (D70); tools/eval_delta.py then writes the stage's per-event mu and sigma for the Delta and pull
# plots. Run dir runs/het_delta/<sweep>__<trial>, predictions $SCRATCH/zz_delta/<sweep>__<trial>__sigma_stage... .
#     site submit <site> FA scripts/job_het_delta.sh -- runs/<sweep>/trial_N [runs/<sweep>/trial_M ...]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
# --shared (D82, the user: "yes, use the shared HPs"): the cell at the het sweeps' shared HPs instead of its own (D75,
# fig:het23): each probe takes its scratch het sweep's best point; run dirs runs/het_shared/, predictions
# $SCRATCH/zz_delta_hetshared/
# --shared-mumse (the user, 2026-10-22): the same at the shared HPs, the checkpoint selected on the mu MSE instead of
# the NLL (the sigma stage then calibrates sigma from that checkpoint); runs/het_shared_mumse/, $SCRATCH/zz_delta_hetshared_mumse/
SHARED=0; ROOT=het_delta; EVAL="$SCRATCH/zz_delta_het"; SEL=nll
[ "$1" = --shared ] && { SHARED=1; ROOT=het_shared; EVAL="$SCRATCH/zz_delta_hetshared"; shift; }
[ "$1" = --shared-mumse ] && { SHARED=1; ROOT=het_shared_mumse; EVAL="$SCRATCH/zz_delta_hetshared_mumse"; SEL=mu_mse; shift; }
# --shared-mumse-clean: the same into a fresh root, for the 8 cells two duplicate jobs trained at once (2026-10-22)
[ "$1" = --shared-mumse-clean ] && { SHARED=1; ROOT=het_shared_mumse_clean; EVAL="$SCRATCH/zz_delta_hetshared_mumse_clean"; SEL=mu_mse; shift; }
shared_hps() {   # the probe of a cell's sweep name -> its HP overrides
    local ee="training.regularization_lambda=5.73461e-08 training.cosanneal_warmup_frac=0.05198803
              training.cosanneal_eta_min=1.353762e-09 ema=true training.ema_decay=0.992479 training.lr=0.002921688"
    local uu="training.regularization_lambda=1.191642e-09 training.cosanneal_warmup_frac=0.06418264
              training.cosanneal_eta_min=6.52542e-10 ema=true training.ema_decay=0.9953661"
    case "$1" in
        *_ee_ddbarg_*|*_ee_ttbarg_*) echo $ee ;;
        *_uubar_Zggg_*) echo $uu training.lr=0.002221238 ;;
        *_uubar_Zgg_*)  echo $uu training.lr=0.0007033921 ;;
        *) echo "no shared HPs for $1" >&2; return 1 ;;
    esac
}
rc=0
for T in "$@"; do
    T="${T%/}"
    OUT="$PROJECT_DIR/runs/$ROOT/$(basename $(dirname $T))__$(basename $T)"
    HP=""; [ $SHARED = 1 ] && { HP=$(shared_hps "$(basename $(dirname $T))") || { rc=1; continue; }; }
    if [ ! -e "$OUT/sigma_stage/models" ]; then
        test -e "$OUT" && mv "$OUT" "$OUT.partial_$SLURM_JOB_ID"    # an earlier attempt that did not finish: kept aside
        # increment_steps cleared (a DyHPO trial's config carries its fidelity there and it sets the cosine horizon);
        # sign_head off (asserted incompatible with HETEROSC; a signed probe keeps its signed-log target)
        # a cell copied from another site carries that site's checkout in its paths: re-rooted onto this one
        RR=$(python - "$PROJECT_DIR/$T/config.yaml" "$PROJECT_DIR" <<'PY'
import re, sys
from omegaconf import OmegaConf
c = OmegaConf.load(sys.argv[1]); here = sys.argv[2]
def rr(p):
    return re.sub(r"^.*?/Foundational_Amplitudes/", here + "/", str(p))
out = [f"base_dir={here}"]
if c.data.get("processes_file"):
    out.append(f"data.processes_file={rr(c.data.processes_file)}")
if c.get("fine_tune") is not None and c.fine_tune.get("pretrained_path"):
    out.append(f"fine_tune.pretrained_path={rr(c.fine_tune.pretrained_path)}")
print(" ".join(out))
PY
)
        python run.py --config-path "$PROJECT_DIR/$T" --config-name config run_dir="$OUT" run_name=null warm_start_idx=null $RR \
            training.is_dyhpo_run=false training.increment_steps=null training.result_path="$OUT/result.json" plot=true \
            training.loss=HETEROSC training.heterosc_beta=0.0 ++training.heterosc_select=$SEL \
            ++model.net.sigma_after_pool=true ++model.net.detach_sigma_backbone=false \
            training.sign_head=false ++training.track_train_eval=true evaluation.train_subsample=2000 $HP || { rc=1; continue; }
    fi
    python tools/eval_delta.py --out-dir "$EVAL" "$OUT/sigma_stage" || rc=1
done
exit $rc
