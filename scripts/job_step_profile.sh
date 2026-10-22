#!/bin/bash
#SBATCH --job-name=step_profile
#SBATCH --cpus-per-task=5
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=step_profile_%j.out
#SBATCH --error=step_profile_%j.out
#
# Why a catalog step costs 0.45 s on a V100 against 0.35 s for rung 9 (D79, the user: "I'd like to understand what's
# making the steps take longer"). Short timed runs, all in this one job so all on the same GPU, each 500 steps at
# batch 16384 with LLOCA_PROFILE_STEP=1 (the data-wait vs forward/backward split, base_experiment.train), one
# validation, no post-training evaluation, no sigma stage. Variants, each one change from its base:
#   cat          the catalog's best trial as it ran (runs/dancat_het_pre/trial_0058: diagrams, memmap, het loss)
#   cat_nodiag   ... without the diagram encoder
#   cat_19       ... on 19 catalog processes drawn at random (seed 0): the process count alone
#   r9           rung 9 as tp3_jaff_r9 ran (no diagrams, in-memory data, MSE)
#   r9_diag      ... with the catalog's diagram encoder (llm4lhc, width 64)
#   r9_memmap    ... with memory-mapped data
#   r9_het       ... with the het loss
# Prints one PROFILE line per variant: the Finished-training line (avg s/iter) and the profiler's split.
#   site submit ccin2p3 FA scripts/job_step_profile.sh --hdr=--mem=40G
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
export LLOCA_PROFILE_STEP=1
OUT="$SCRATCH/zz_step_profile_$SLURM_JOB_ID"; mkdir -p "$OUT"
nvidia-smi --query-gpu=name --format=csv,noheader
SUB="$OUT/catalog_19.yaml"
python - "$SUB" <<'PY'
import random, sys, yaml
r = yaml.safe_load(open("recipes/catalog_daniel_train.yaml"))
random.Random(0).shuffle(r["processes"]); r["processes"] = r["processes"][:19]
yaml.safe_dump(r, open(sys.argv[1], "w")); print("cat_19:", [p["name"] for p in r["processes"]])
PY
COMMON=(warm_start_idx=null run_name=null base_dir="$PROJECT_DIR" training.is_dyhpo_run=false training.increment_steps=null
        training.iterations=500 training.validate_frac=1.0 evaluate=false ++training.heterosc_sigma_stage_steps=0
        ++training.track_train_eval=false)
HET=(training.loss=HETEROSC training.heterosc_beta=0.0 ++training.heterosc_select=nll ++model.net.sigma_after_pool=true
     ++model.net.detach_sigma_backbone=false training.sign_head=false)
run() {   # name, config dir, overrides...
    local name="$1" dir="$2"; shift 2
    python run.py --config-path "$PROJECT_DIR/$dir" --config-name config run_dir="$OUT/$name" "${COMMON[@]}" "$@" \
        > "$OUT/$name.log" 2>&1
    local fin prof
    fin=$(grep -a -m1 "Finished training" "$OUT/$name.log" | sed 's/.*Finished training: //')
    prof=$(grep -a "PROFILE summary" "$OUT/$name.log" | tail -1 | tr '\n' ' ')
    echo "PROFILE $name | ${fin:-FAILED, see $OUT/$name.log} | $prof"
    [ -n "$fin" ] || tail -5 "$OUT/$name.log"
}
C=runs/dancat_het_pre/trial_0058
R=runs/tp3_jaff_r9/trial_0073
run cat        "$C"
run cat_nodiag "$C" model.use_diagrams=false
run cat_19     "$C" data.processes_file="$SUB"
run r9         "$R"
run r9_diag    "$R" model.use_diagrams=true ++model.diagram_encoder_impl=llm4lhc ++model.diagram_encoder_llm4lhc.hidden_size=64
run r9_memmap  "$R" ++data.memmap=true
run r9_het     "$R" "${HET[@]}"
