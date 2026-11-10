#!/bin/bash
#SBATCH --job-name=dsave
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# pretrain_data_saving: one run of a held-out catalog_v4 target on its first D = 10^(k/2) train events, on the
# model and inputs of the parent pretraining (its saved config.yaml is the base config, so every setting not
# overridden here is the parent's), from scratch or fine-tuned from the parent.
#     site submit <site|auto> FA scripts/job_data_saving.sh -- <target> <k> <steps> [scratch|ft] [key=value ...]
# <target> names recipes/catalog_v4_datasaving_<target>.yaml (one process). training.lr defaults to the parent's lr
# law at <steps>, 7.82114e-3 (steps/17056)^-0.55 (the trial-1262 law of job_catalog_v3_short.sh LONG_HP), times the
# transfer study's fitted D factor (sweep/gen_transfer_pilot.lr_surface's D part); a training.lr=... argument wins.
# The batch is the parent's 16384, which experiment.py caps at D/2 (the transfer study's B = min(16384, D/2)).
# warm_start_idx=null: the parent (a sweep trial) saved warm_start_idx 0, which would reload a run dir of the same
# name. The checkpoint is selected on the validation mu MSE (training.heterosc_select=mu_mse, as D85/D89), then the sigma
# stage; the readout is the MSE of ln|M|^2 of the mean, whose validation curve (val_mse_curve, one point per
# validate_every_n_steps), best_step and val_curve go to <run_dir>/result.json (training.result_path; null in the parent
# config, so without it no curve of the mu MSE is kept). Run dir runs/data_saving/dsave_<mode>_<target>_d<k>_t<steps>[_<tag>] (tag=<name> argument).
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
TGT="$1"; K="$2"; T="$3"; shift 3
MODE=scratch; case "$1" in scratch|ft) MODE="$1"; shift ;; esac
PARENT="${DS_PARENT:-runs/catalog_v4_long/best_cov_v4_long_hpo8_cc_68224_trial_1262}"
RECIPE="$PROJECT_DIR/recipes/catalog_v4_datasaving_${TGT}.yaml"
[ -e "$RECIPE" ] || { echo "no recipe $RECIPE"; exit 2; }
[ -e "$PROJECT_DIR/$PARENT/config.yaml" ] || { echo "no parent config $PARENT/config.yaml"; exit 2; }
[[ "$K" =~ ^([2-9]|10)$ ]] && [[ "$T" =~ ^[0-9]+$ ]] || { echo "k in 2..10, steps an integer"; exit 2; }
D=$(python -c "print(round(10 ** ($K / 2)))")
TAG=; ARGS=(); for x in "$@"; do case "$x" in tag=*) TAG="_${x#tag=}";; seed=*) TAG="${TAG}_seed${x#seed=}"; ARGS+=("$x");; *) ARGS+=("$x");; esac; done
case "$TGT" in *_nlo|*_loop) [ "$K" -le 9 ] || { echo "one-loop pools hold 5e4 train events: k <= 9"; exit 2; } ;; esac
case " ${ARGS[*]} " in
    *" training.lr="*) ;;
    *) LR=$(PYTHONPATH=. python -c "
import sys; sys.argv = ['x']
from sweep.gen_transfer_pilot import D_FACTOR
import numpy as np
t, D = $T, $D
Ns = np.array(sorted(D_FACTOR)); B = np.array([D_FACTOR[n] for n in Ns])
f = float(np.exp(np.interp(np.log(np.clip(D, Ns.min(), Ns.max())), np.log(Ns), B)))
print(f'{7.82114e-3 * (t / 17056) ** -0.55 * f:.6g}')") || exit 2
       ARGS+=("training.lr=$LR") ;;
esac
FT=(fine_tune.pretrained_path=null)
if [ "$MODE" = ft ]; then
    PT="$PROJECT_DIR/$PARENT/models/model_run0_best.pt"; [ -e "$PT" ] || PT="$PT.gz"
    FT=(fine_tune.pretrained_path="$PT" fine_tune.target_stats=own fine_tune.offshell_stats=parent)
fi
OUT="$PROJECT_DIR/runs/data_saving/dsave_${MODE}_${TGT}_d${K}_t${T}${TAG}"
echo "DATA_SAVING target=$TGT k=$K D=$D steps=$T mode=$MODE parent=$PARENT args: ${ARGS[*]}"
python run.py --config-path "$PROJECT_DIR/$PARENT" --config-name config run_dir="$OUT" run_name=null \
    data.processes_file="$RECIPE" data.train_subsample="$D" data.eval_subsample=null data.phase_space_on_pool=true \
    data.memmap=false training.iterations="$T" training.heterosc_select=mu_mse evaluation.train_subsample=2000 seed=42 warm_start_idx=null training.result_path="$OUT/result.json" "${FT[@]}" "${ARGS[@]}"
RC=$?
echo "DATA_SAVING_DONE rc=$RC $OUT"
# the supervisor's clean-end marker (~/.config/ccorch/projects.toml end_markers), only on success
[ "$RC" -eq 0 ] && echo "RUN_DONE $OUT"
exit $RC
