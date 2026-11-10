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
# The parent config runs data.memmap=true, so each recipe needs its memmap cache built first on CPU (a GPU run
# without it refuses at data loading): scripts/job_mmap_prebuild.sh --config-path <parent> --config-name config
#   data.processes_file=recipes/catalog_v4_kit_pretrain_<recipe>.yaml fine_tune.pretrained_path=null
#   training.iterations=68224 training.is_dyhpo_run=false, and this job submitted with --dependency=afterok:<its job>
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
if [ "${KIT_RESUME:-}" = 1 ]; then
    # KIT_RESUME=1 (site submit --env, with --hdr=--dependency=afterany:<the first job>): continue a pretraining its
    # time limit stopped (the first two ran 0.94 s/step on V100, 17.9 h for 68224 steps against a 12 h limit). It
    # starts from the latest valid checkpoint with optimizer, scheduler and step (model_run0_best.pt, written at each
    # validation that improved, every 1364 steps; else model_run0_current.pt), copied to model_run0.pt for run.py's
    # warm start (warm_start_idx=0: run_idx 1, the same run dir), and trains the remaining steps of the same cosine
    # (training.increment_steps=68224 keeps T_max and the warmup; the scheduler's last_epoch is restored). A first job
    # that finished (model_run0.pt without resumed_from.txt) or a finished resume (model_run1.pt) ends at once.
    # training.resume_seed_best makes model_run1_best.pt the best over both segments (A_ft takes the last segment's).
    M="$OUT/models"
    if { [ -e "$M/model_run0.pt" ] && [ ! -e "$M/resumed_from.txt" ]; } || [ -e "$M/model_run1.pt" ]; then
        echo "KIT_PRETRAIN_RESUME: $OUT already finished, nothing to resume"; echo "RUN_DONE $OUT"; exit 0
    fi
    PICK=$(python - "$M" <<'PYEOF'
import os, sys, torch
best = None
for f in ("model_run0_best.pt", "model_run0_current.pt"):
    try:
        ck = torch.load(os.path.join(sys.argv[1], f), map_location="cpu", weights_only=False)
        done = int(ck["scheduler"]["last_epoch"])
        assert ck.get("optimizer") and done > 0
    except Exception as e:
        print(f"KIT_PRETRAIN_RESUME: {f} not usable ({type(e).__name__}: {e})", file=sys.stderr)
        continue
    if best is None or done > best[1]:
        best = (f, done)
if best is None:
    sys.exit(1)
print(best[0], best[1])
PYEOF
) || { echo "KIT_PRETRAIN_RESUME: no usable checkpoint in $M"; exit 2; }
    read -r F DONE <<< "$PICK"
    REM=$((68224 - DONE))
    [ "$REM" -gt 0 ] || { echo "KIT_PRETRAIN_RESUME: $F has $DONE steps, nothing left"; exit 2; }
    cp "$M/$F" "$M/model_run0.pt" && echo "$F last_epoch=$DONE remaining=$REM $(date -u +%FT%TZ)" > "$M/resumed_from.txt"
    echo "KIT_PRETRAIN_RESUME from $F at step $DONE: $REM steps left, args: ${ARGS[*]}"
    python run.py --config-path "$OUT" --config-name config run_dir="$OUT" warm_start_idx=0 run_idx=0 \
        training.iterations="$REM" training.increment_steps=68224 training.is_dyhpo_run=false \
        ++training.resume_seed_best=true "${ARGS[@]}"
    RC=$?
    echo "KIT_PRETRAIN_DONE rc=$RC $OUT"
    [ "$RC" -eq 0 ] && echo "RUN_DONE $OUT"
    exit $RC
fi
echo "KIT_PRETRAIN recipe=$RECIPE parent=$PARENT args: ${ARGS[*]}"
python run.py --config-path "$PROJECT_DIR/$PARENT" --config-name config run_dir="$OUT" run_name=null \
    data.processes_file="$RECIPE" fine_tune.pretrained_path=null warm_start_idx=null training.iterations=68224 training.is_dyhpo_run=false "${ARGS[@]}"
RC=$?
echo "KIT_PRETRAIN_DONE rc=$RC $OUT"
# the supervisor's clean-end marker (~/.config/ccorch/projects.toml end_markers), only on success
[ "$RC" -eq 0 ] && echo "RUN_DONE $OUT"
exit $RC
