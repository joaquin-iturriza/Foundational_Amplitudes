#!/bin/bash
#SBATCH --job-name=kit
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# kit_benchmark: one run on a KIT hard phase-space dataset (tools/kit_hardps.py), float64 end to end, then its
# benchmark scores (tools/kit_predict.py: predictions on every hard_test stratum and bulk_test, the benchmark's own
# evaluate.py).
#     site submit <site|auto> FA scripts/job_kit.sh -- <dataset> <n_train> <steps> <mode> [key=value ...]
# KIT_FROM (B, B_bulk) and KIT_PARENT (A_ft) reach the job only through `site submit --env KIT_FROM=<run dir>`: a
# shell variable set before `site submit` is not exported to the job (12 B jobs failed so, 2026-11-10).
# <dataset>  kit_<file>__<target>, e.g. kit_ee_uug__me2, kit_ee_tt__virt_finite
# <n_train>  the first n_train events of bulk_train; validation is the last 1e5 of bulk_train (bulk-only early
#            stopping), test the 2e5 of bulk_test
# <mode>     A        protocol A from scratch, on the model and inputs of the catalog pretraining (its saved
#                     config.yaml is the base config, so every setting not overridden here is the parent's);
#                     training.lr is the parent's lr law at <steps>, 7.82114e-3 (steps/17056)^-0.55, times the
#                     transfer study's D factor (as scripts/job_data_saving.sh); a training.lr=... argument wins
#            A_ft     protocol A fine-tuned from KIT_PARENT (a pretraining with the target's family held out), at the A
#                     runs' lr (fine_tune.lr_scale=1.0; the parent config's 0.1 would change the lr with the start)
#            B        protocol B: continued from the protocol A run KIT_FROM (its config is the base, its best
#                     weights the start, its target and off-shellness stats kept) with every hard_train stratum
#                     added to the train split (dataset kit_<file>__<target>__hard: the first 90% of each stratum;
#                     validation is bulk plus the last 10%, so the checkpoint sees the hard fit)
#            B_bulk   B's matched control: the same continuation from KIT_FROM (same steps, lr and restart) on the
#                     bulk only
# The checkpoint is selected on the validation mu MSE (training.heterosc_select=mu_mse), then the sigma stage.
# Run dir runs/kit/kit_<mode>_<file>_<target>_n<n_train>_t<steps>[_<tag>] (tag=<name> argument).
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
DS="$1"; N="$2"; T="$3"; MODE="$4"; shift 4
[[ "$DS" =~ ^kit_[A-Za-z0-9_]+__[A-Za-z0-9_]+$ ]] && [[ "$DS" != *__hard ]] || { echo "dataset kit_<file>__<target>"; exit 2; }
[[ "$N" =~ ^[0-9]+$ ]] && [[ "$T" =~ ^[0-9]+$ ]] || { echo "n_train and steps are integers"; exit 2; }
case "$MODE" in A|A_ft|B|B_bulk) ;; *) echo "mode A, A_ft, B or B_bulk"; exit 2 ;; esac
PARENT="${KIT_PARENT:-runs/catalog_v4_long/best_cov_v4_long_hpo8_cc_68224_trial_1262}"
# KIT_NVAL / KIT_NTEST / KIT_NO_SCORE=1 only for a CPU dry run of the setup on a login node
NVAL="${KIT_NVAL:-100000}"; NTEST="${KIT_NTEST:-200000}"
TAG=; ARGS=(); for x in "$@"; do case "$x" in tag=*) TAG="_${x#tag=}";; seed=*) TAG="${TAG}_seed${x#seed=}"; ARGS+=("$x");; *) ARGS+=("$x");; esac; done
STEM="${DS#kit_}"; FILE="${STEM%%__*}"; TGT="${STEM#*__}"
OUT="$PROJECT_DIR/runs/kit/kit_${MODE}_${FILE}_${TGT}_n${N}_t${T}${TAG}"
NAME="$DS"; [ "$MODE" = B ] && NAME="${DS}__hard"
case "$FILE" in *tt*) SQ="[346, 50000]" ;; *) SQ="[1000, 1000]" ;; esac
# the recipe lives beside the run dir, not in it: run.py refuses an existing run dir (a fresh start, never a warm one)
mkdir -p "$PROJECT_DIR/runs/kit/recipes"
RECIPE="$PROJECT_DIR/runs/kit/recipes/$(basename "$OUT").yaml"
printf 'processes:\n- name: %s\n  sqrts: %s\n  n_train: %s\n  n_val: %s\n  n_test: %s\n' "$NAME" "$SQ" "$N" "$NVAL" "$NTEST" > "$RECIPE"
if [ "$MODE" = B ] || [ "$MODE" = B_bulk ]; then
    [ -n "$KIT_FROM" ] && [ -e "$PROJECT_DIR/$KIT_FROM/config.yaml" ] || { echo "B needs KIT_FROM=<protocol A run dir>"; exit 2; }
    BASE="$PROJECT_DIR/$KIT_FROM"
    PT="$BASE/models/model_run0_best.pt"; [ -e "$PT" ] || PT="$PT.gz"
    FT=(fine_tune.pretrained_path="$PT" fine_tune.target_stats=parent fine_tune.offshell_stats=parent)
else
    BASE="$PROJECT_DIR/$PARENT"
    [ -e "$BASE/config.yaml" ] || { echo "no parent config $PARENT/config.yaml"; exit 2; }
    case " ${ARGS[*]} " in
        *" training.lr="*) ;;
        *) LR=$(PYTHONPATH=. python -c "
import sys; sys.argv = ['x']
from sweep.gen_transfer_pilot import D_FACTOR
import numpy as np
t, D = $T, $N
Ns = np.array(sorted(D_FACTOR)); B = np.array([D_FACTOR[n] for n in Ns])
f = float(np.exp(np.interp(np.log(np.clip(D, Ns.min(), Ns.max())), np.log(Ns), B)))
print(f'{7.82114e-3 * (t / 17056) ** -0.55 * f:.6g}')") || exit 2
           ARGS+=("training.lr=$LR") ;;
    esac
    FT=(fine_tune.pretrained_path=null)
    if [ "$MODE" = A_ft ]; then
        # the parent's last segment (a pretraining resumed after its time limit, KIT_RESUME, writes model_run1_best.pt)
        PT=; for k in 9 8 7 6 5 4 3 2 1 0; do for f in "model_run${k}_best.pt" "model_run${k}_best.pt.gz"; do
            [ -z "$PT" ] && [ -e "$PROJECT_DIR/$PARENT/models/$f" ] && PT="$PROJECT_DIR/$PARENT/models/$f"; done; done
        [ -n "$PT" ] || { echo "A_ft: no model_run*_best.pt in $PARENT/models"; exit 2; }
        FT=(fine_tune.pretrained_path="$PT" fine_tune.target_stats=own fine_tune.offshell_stats=parent fine_tune.lr_scale=1.0)
    fi
fi
echo "KIT dataset=$NAME n_train=$N steps=$T mode=$MODE base=$BASE args: ${ARGS[*]}"
if [ -n "$KIT_SCORE_ONLY" ]; then
    # re-score a run that trained but whose scoring failed (same arguments name the same run dir); no training
    [ -e "$OUT/config.yaml" ] || { echo "KIT_SCORE_ONLY: no trained run at $OUT"; exit 2; }
    python tools/kit_predict.py "$OUT"; RC=$?
    echo "KIT_DONE rc=$RC $OUT"; [ "$RC" -eq 0 ] && echo "RUN_DONE $OUT"; exit $RC
fi
python run.py --config-path "$BASE" --config-name config run_dir="$OUT" run_name=null \
    data.source=recipes data.processes_file="$RECIPE" data.train_subsample=null data.eval_subsample=null \
    data.memmap=false training.dtype=float64 training.iterations="$T" training.heterosc_select=mu_mse \
    evaluation.train_subsample=2000 seed=42 warm_start_idx=null training.increment_steps=null "${FT[@]}" "${ARGS[@]}"
RC=$?
if [ $RC -eq 0 ] && [ -z "$KIT_NO_SCORE" ]; then
    python tools/kit_predict.py "$OUT"
    RC=$?
fi
echo "KIT_DONE rc=$RC $OUT"
# the supervisor's clean-end marker (~/.config/ccorch/projects.toml end_markers), only on success
[ "$RC" -eq 0 ] && echo "RUN_DONE $OUT"
exit $RC
