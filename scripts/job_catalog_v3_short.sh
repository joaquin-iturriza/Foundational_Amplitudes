#!/bin/bash
#SBATCH --job-name=catalog_v3_short
#SBATCH --cpus-per-task=5
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=catalog_v3_short_%j.out
#SBATCH --error=catalog_v3_short_%j.out
#
# D115 (the user, 2026-10-25: "switch to the new catalog now that it's done, run the pretrainings there"): catalog_v3's
# short pretrainings, the short setup of Daniel's catalog round (scripts/job_catalog_norm.sh short: hp58's configuration and
# HPs, 1e4 train and 200 val events per process, 5 epochs, in memory) on recipes/catalog_v3_train.yaml (1397 pools,
# 5 epochs: 4264 steps), then every generated held-out axis read zero-shot (H7, 100 TeV and pT >= 30 GeV, is not generated) (tools/zero_shot_parent.py, 2000 val
# events per process; the one-loop H5 pools on log|M|, --signed-magnitude). Arms:
#   B      one standardization shared by all processes, log|M|^2 in GeV (the baseline)
#   Q      the adopted target: dimensionless (s_hat per event, dimensionless inputs), the coupling orders, the emission
#          scales (data.target_physics_dim_ref=qperp), the arXiv:2606.23791 diagram encoder (DlessInOrdQ)
#   Qc     Q with the per-process context vector (model.context_dim 16, dropout 0.5; D105)
#   Qtwin  Q trained on the removal twin (recipes/catalog_v3_twin_train.yaml, 1246 pools: none with a tau or nu_tau in any
#          layer, none built on the quartic gauge family), read besides the axes on H3p (the 136 pools with the never-seen
#          tau or nu_tau) and H4 (the 15 quartic-gauge pools); 5 epochs of its own set (3803 steps)
# Every arm carries the sign head for catalog_v3's signed one-loop pools (the signed A/B, docs/results.tex) and so trains
# on the MSE with the arithmetic mean over processes (the sign head is not wired into the het loss): one loss for all.
#   site submit ccin2p3 FA scripts/job_catalog_v3_short.sh --hdr=--mem=48G -- B|Q|Qc|Qtwin [smoke|eval]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
ARM="${1:?B, Q, Qc or Qtwin}"; MODE="${2:-}"
SRC=runs/dancat_het_pre/trial_0058
TRAIN="$PROJECT_DIR/recipes/catalog_v3_train.yaml"
COMMON=(training.loss=MSE ++training.loss_aggregation=mean ++training.sign_head=true)
DIM=(++data.target_physics_factor=true ++data.target_physics_dim_only=true ++data.target_physics_s0=1.0
     ++data.dimensionless_inputs=true ++data.target_physics_orders=true ++data.target_physics_dim_ref=qperp
     ++data.shared_standardization=true ++data.affine_multiplicity_std=false)
case "$ARM" in
    B)     NORM=(++data.shared_standardization=true ++data.affine_multiplicity_std=false) ;;
    Q)     NORM=("${DIM[@]}") ;;
    Qc)    NORM=("${DIM[@]}" ++model.context_dim=16 ++model.context_dropout=0.5) ;;
    Qtwin) NORM=("${DIM[@]}"); TRAIN="$PROJECT_DIR/recipes/catalog_v3_twin_train.yaml" ;;
    *) echo "unknown arm $ARM"; exit 1 ;;
esac
OUT="$PROJECT_DIR/runs/catalog_v3_short/$ARM"; PRED="$SCRATCH/zz_v3_short_$ARM"
# 5 epochs of 1e4 events per process at batch 16384, from the recipe in use (1397 pools: 4264 steps; the twin's 1246: 3803)
ITERS=$(python -c "import math, sys, yaml; print(math.ceil(len(yaml.safe_load(open(sys.argv[1]))['processes']) * 1e4 / 16384 * 5))" "$TRAIN")
EXTRA=(training.validate_frac=0.02 training.iterations=$ITERS ++data.memmap=false ++data.train_subsample=10000
       ++data.eval_subsample=200 evaluation.train_subsample=2000 data.processes_file="$TRAIN")
AXES=(H1 H2 H2p5 H3 H5 H6); [ "$ARM" = Qtwin ] && AXES+=(H3p H4)
if [ "$MODE" = smoke ]; then
    OUT="$SCRATCH/zz_v3_smoke_${ARM}_$SLURM_JOB_ID"; PRED="$OUT/preds"
    SUB="$OUT.recipe.yaml"
    python - "$TRAIN" "$SUB" <<'PY'
import random, sys, yaml
r = yaml.safe_load(open(sys.argv[1])); ps = r["processes"]
loops = [p for p in ps if p["name"].endswith(("_nlo", "_loop"))]
random.Random(0).shuffle(ps); r["processes"] = ps[:16] + loops[:3]
yaml.safe_dump(r, open(sys.argv[2], "w"))
PY
    EXTRA=(training.iterations=40 training.validate_frac=0.5 evaluate=false data.processes_file="$SUB"
           ++data.memmap=false ++data.train_subsample=2000 ++data.eval_subsample=200)
    AXES=(H1 H5)
fi
if [ "$MODE" != eval ]; then
    if [ ! -e "$OUT/models/model_run0_best.pt" ]; then
        test -e "$OUT" && mv "$OUT" "$OUT.partial_$SLURM_JOB_ID"
        python run.py --config-path "$PROJECT_DIR/$SRC" --config-name config run_dir="$OUT" run_name=null warm_start_idx=null \
            base_dir="$PROJECT_DIR" training.is_dyhpo_run=false training.increment_steps=null \
            training.result_path="$OUT/result.json" plot=true "${COMMON[@]}" "${NORM[@]}" "${EXTRA[@]}" || exit 1
    fi
fi
ls "$OUT"/models/model_run0_best.pt* >/dev/null 2>&1 || { echo "no best checkpoint in $OUT"; exit 1; }
R="$SCRATCH/zz_v3_heldout_recipes"
for A in "${AXES[@]}"; do   # one recipe per process, keeping the axis file's top-level keys (sampling: mixture)
    mkdir -p "$R/$A" "$PRED/$A"
    python - "$R/$A" "recipes/catalog_v3_heldout_$A.yaml" <<'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[2]))
top = {k: v for k, v in r.items() if k != "processes"}
for p in r["processes"]:
    yaml.safe_dump({**top, "processes": [p]}, open(f"{sys.argv[1]}/{p['name']}.yaml", "w"))
PY
    P=("$R/$A"/*.yaml); [ "$MODE" = smoke ] && P=("${P[@]:0:2}")
    SM=(); [ "$A" = H5 ] && SM=(--signed-magnitude)
    python tools/zero_shot_parent.py --parent "$OUT" --save-dir "$PRED/$A" "${SM[@]}" "${P[@]}" | sed -u "s/^/$A /"
    [ "${PIPESTATUS[0]}" = 0 ] || exit 1
    # H5's sign-changing pools: the sign head's own scoring too (magnitude and wrong-sign share, the signed A/B's metrics)
    if [ "$A" = H5 ]; then
        python tools/signed_ab_score.py --run "$OUT" --arm head "${P[@]}" | tee "$PRED/H5_signed.txt"
        [ "${PIPESTATUS[0]}" = 0 ] || exit 1
    fi
done
echo "V3_SHORT_DONE $ARM"
