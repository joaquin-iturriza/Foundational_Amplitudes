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
rc=0
for T in "$@"; do
    T="${T%/}"
    OUT="$PROJECT_DIR/runs/het_delta/$(basename $(dirname $T))__$(basename $T)"
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
            training.loss=HETEROSC training.heterosc_beta=0.0 ++training.heterosc_select=nll \
            ++model.net.sigma_after_pool=true ++model.net.detach_sigma_backbone=false \
            training.sign_head=false ++training.track_train_eval=true evaluation.train_subsample=2000 || { rc=1; continue; }
    fi
    python tools/eval_delta.py --out-dir "$SCRATCH/zz_delta_het" "$OUT/sigma_stage" || rc=1
done
exit $rc
