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
# 5 epochs: 4264 steps), then every held-out axis read zero-shot (tools/zero_shot_parent.py, 2000 val
# events per process; the one-loop H5 pools on log|M|, --signed-magnitude). Arms (named for what they are; the
# first round's runs were started as B, Q, Qc, Qtwin and their directories renamed to these):
#   baseline               one standardization shared by all processes, log|M|^2 in GeV
#   dimensionless          the adopted target: |M|^2 made dimensionless per event (s_hat), divided by its coupling
#                          orders, with each extra particle's emission scale; dimensionless inputs; the arXiv:2606.23791
#                          diagram encoder (Daniel's catalog round: DlessInOrdQ)
#   dimensionless_context  the same with a learned per-process context vector (16, zeroed on half the events)
#   dimensionless_pp       the same with every hadronic tree class trained on 13 TeV pp events (the pp study)
#   dimensionless_twin     the same trained on the removal twin (recipes/catalog_v3_twin_train.yaml, 1246 pools: none with a tau or nu_tau in any
#          layer, none built on the quartic gauge family), read besides the axes on H3p (the 136 pools with the never-seen
#          tau or nu_tau) and H4 (the 15 quartic-gauge pools); 5 epochs of its own set (3803 steps)
# Every arm carries the sign head for catalog_v3's signed one-loop pools (the signed A/B, docs/results.tex) and so trains
# on the MSE with the arithmetic mean over processes (the sign head is not wired into the het loss): one loss for all.
#   dimensionless_shat     the same with s_hat alone as the target's reference scale (the shape proxy's candidate)
#   dimensionless_hybrid   the s_hat target plus a per-process constant from the events' momenta (the hybrid reference scale)
#   dimensionless_blend050, dimensionless_blend025   the hybrid target plus 0.5 / 0.25 of the emission-scale term minus its process mean
#   dimensionless_muon     the dimensionless arm trained with Muon (decoupled weight decay 0.1, update RMS matched to AdamW's)
#                          on muP's 32 hidden matrices and AdamW as before on the rest (optim/muon.py; the Muon test, D144/D146)
#   dimensionless_seed2, dimensionless_muon_seed2   the same two with a second seed (model init and batch order, seed 43;
#                          the pools and their order unchanged)
# Mode smoke also fine-tunes the smoke parent on one held-out probe, 20 steps, with AdamW and with Muon (tools/catalog_ft.py).
#   site submit ccin2p3 FA scripts/job_catalog_v3_short.sh --hdr=--mem=48G -- <arm> [smoke|eval]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
ARM="${1:?baseline, baseline_sign_pairs, baseline_sign_near, baseline_more_signed, dimensionless, dimensionless_shat, dimensionless_hybrid, dimensionless_blend050, dimensionless_blend025, dimensionless_context, dimensionless_twin, dimensionless_pp, dimensionless_muon, dimensionless_seed2 or dimensionless_muon_seed2}"; MODE="${2:-}"
SRC=runs/dancat_het_pre/trial_0058
TRAIN="$PROJECT_DIR/recipes/catalog_v3_train.yaml"
# the sign head for the signed one-loop pools; the diagram encoder's edge labels for the BSM layer's particles
COMMON=(training.loss=MSE ++training.loss_aggregation=mean ++training.sign_head=true
        ++model.diagram_encoder_llm4lhc.bsm_labels=true)
DIM=(++data.target_physics_factor=true ++data.target_physics_dim_only=true ++data.target_physics_s0=1.0
     ++data.dimensionless_inputs=true ++data.target_physics_orders=true ++data.target_physics_dim_ref=qperp
     ++data.shared_standardization=true ++data.affine_multiplicity_std=false)
case "$ARM" in
    baseline)              NORM=(++data.shared_standardization=true ++data.affine_multiplicity_std=false) ;;
    dimensionless)         NORM=("${DIM[@]}") ;;
    dimensionless_context) NORM=("${DIM[@]}" ++model.context_dim=16 ++model.context_dropout=0.5) ;;
    # the shape proxy's candidate (docs/results.tex, par. The shape proxy on catalog_v3; D99): the same dimensionless
    # inputs and orders, the target's reference scale s_hat alone in place of the emission scales
    dimensionless_shat)    NORM=("${DIM[@]/target_physics_dim_ref=qperp/target_physics_dim_ref=shat}") ;;
    # the hybrid reference scale (docs/results.tex, par. The hybrid reference scale, a design; D129): the s_hat target
    # plus one constant per process, c_p = the mean over its own events of the emission scales minus (N - 4) log s_hat
    # (experiment._physics_hybrid_cp), so the shape within a process is the s_hat arm's and the class mean the
    # emission-scale arm's; everything else as the dimensionless arm
    dimensionless_hybrid)  NORM=("${DIM[@]/target_physics_dim_ref=qperp/target_physics_dim_ref=shat}" ++data.target_physics_hybrid=true) ;;
    # the blend of the hybrid and the emission scales (the user 2026-10-25, "run those two pretrainings blending
    # emission scales"): the hybrid target plus lambda times the per-event emission-scale term minus its process mean
    # (physics_scales.hybrid_blend); lambda 0 is the hybrid arm, 1 the dimensionless arm's target
    dimensionless_blend050) NORM=("${DIM[@]/target_physics_dim_ref=qperp/target_physics_dim_ref=shat}" ++data.target_physics_hybrid=true
                                  ++data.target_physics_hybrid_blend=0.5) ;;
    dimensionless_blend025) NORM=("${DIM[@]/target_physics_dim_ref=qperp/target_physics_dim_ref=shat}" ++data.target_physics_hybrid=true
                                  ++data.target_physics_hybrid_blend=0.25) ;;
    dimensionless_twin)    NORM=("${DIM[@]}"); TRAIN="$PROJECT_DIR/recipes/catalog_v3_twin_train.yaml" ;;
    # the pp study: the same target trained with every hadronic tree class on 13 TeV MadEvent pp events in place of its
    # shaped window (recipes/catalog_v3_pp_mixed_train.yaml; the other pools as in catalog_v3_train.yaml)
    dimensionless_pp)      NORM=("${DIM[@]}"); TRAIN="$PROJECT_DIR/recipes/catalog_v3_pp_mixed_train.yaml" ;;
    # the Muon test (D144, D146): the dimensionless arm with MuonAdamW at the same lr number (by muP's rule for Muon, its
    # update matches AdamW's at the base width); the second seeds of it and of the AdamW arm
    dimensionless_muon)    NORM=("${DIM[@]}" training.optimizer=MuonAdamW) ;;
    dimensionless_seed2)   NORM=("${DIM[@]}" seed=43) ;;
    dimensionless_muon_seed2) NORM=("${DIM[@]}" training.optimizer=MuonAdamW seed=43) ;;
    # the one-loop sign transfer (docs/results.tex, par. The signed one-loop pools' data: the kinematics decide the
    # sign within a pool, the arms fail it zero-shot), each the baseline with one change, at its HPs:
    #   baseline_sign_pairs  the sign logit also gets a pair term on the invariants the per-pool classifier used
    #                        (model.sign_pair_dim 16 per-particle pair features; wrappers._sign_pair_logit)
    #   baseline_sign_near   the sign loss weighs the events within a factor 10 of zero 10 times the others, each
    #                        pool's mean weight kept (training.sign_near_weight)
    # the factor 10 is the near-zero events' definition in that paragraph; sign_near_weight 10, sign_pair_dim 16 and
    # the pair MLP's 32 hidden units are open (unsourced) settings, one untuned value each: a loss says that value
    # does not help, not that the idea fails
    baseline_sign_pairs)   NORM=(++data.shared_standardization=true ++data.affine_multiplicity_std=false ++model.sign_pair_dim=16) ;;
    # the baseline trained on more sign-changing one-loop pools (recipes/catalog_v3_train_signext.yaml: catalog_v3_train
    # plus the 95 one-loop pools of the coloured 2->3 training trees whose train events take both signs,
    # recipes/make_catalog_v3_train_signext.py), at the baseline's 4264 steps (same compute)
    baseline_more_signed)  NORM=(++data.shared_standardization=true ++data.affine_multiplicity_std=false)
                           TRAIN="$PROJECT_DIR/recipes/catalog_v3_train_signext.yaml"; ITERS_FIXED=4264 ;;
    baseline_sign_near)    NORM=(++data.shared_standardization=true ++data.affine_multiplicity_std=false
                                 ++training.sign_near_weight=10 ++training.sign_near_factor=10) ;;
    *) echo "unknown arm $ARM"; exit 1 ;;
esac
OUT="$PROJECT_DIR/runs/catalog_v3_short/$ARM"; PRED="$SCRATCH/zz_v3_short_$ARM"
# 5 epochs of 1e4 events per process at batch 16384, from the recipe in use (1397 pools: 4264 steps; the twin's 1246: 3803)
ITERS=${ITERS_FIXED:-$(python -c "import math, sys, yaml; print(math.ceil(len(yaml.safe_load(open(sys.argv[1]))['processes']) * 1e4 / 16384 * 5))" "$TRAIN")}
EXTRA=(training.validate_frac=0.02 training.iterations=$ITERS ++data.memmap=false ++data.train_subsample=10000
       ++data.eval_subsample=200 evaluation.train_subsample=2000 data.processes_file="$TRAIN")
AXES=(H1 H2 H2p5 H3 H5 H6 H7E H7c); [ "$ARM" = dimensionless_twin ] && AXES+=(H3p H4)
# mode eval may read a subset of the axes again (V3_AXES="H3p"), the trained model reused
[ "$MODE" = eval ] && [ -n "$V3_AXES" ] && read -ra AXES <<< "$V3_AXES"
# every process of the arm's whole training recipe needs its diagram sidecar (the parent's diagram encoder refuses a
# missing one, the off-shellness silently skips it); checked here before any training, in every mode, so a smoke on
# its 19-pool subset fails on what the full run would (baseline_more_signed's full run died on 95 missing sidecars
# after a passing smoke). Build missing ones with scripts/job_catalog_v3_prep.sh <recipe>.
if [ "$MODE" != eval ]; then
    python - "$TRAIN" <<'PY' || exit 1
import os, sys, yaml
names = sorted({p.get("base", p["name"]) for p in yaml.safe_load(open(sys.argv[1]))["processes"]})
miss = [n for n in names if not os.path.exists(f"data/diagrams/{n}.diagrams.json")]
print(f"V3_SIDECARS {len(names) - len(miss)}/{len(names)} present for {os.path.basename(sys.argv[1])}")
if miss:
    sys.exit(f"V3_SIDECARS missing {len(miss)}: {miss}; run scripts/job_catalog_v3_prep.sh {sys.argv[1]}")
PY
fi
if [ "$MODE" = smoke ]; then
    OUT="$SCRATCH/zz_v3_smoke_${ARM}_$SLURM_JOB_ID"; PRED="$OUT/preds"
    SUB="$OUT.recipe.yaml"
    python - "$TRAIN" "$SUB" <<'PY'
import random, sys, yaml
r = yaml.safe_load(open(sys.argv[1])); ps = r["processes"]
loops = [p for p in ps if p["name"].endswith(("_nlo", "_loop"))]
# the sign-changing one-loop pools first (any role with a negative value), so the sign loss is exercised
import json
au = json.load(open("analysis/catalog_v3/pool_audit.json"))["pools"]
au.update(json.load(open("analysis/catalog_v3/pool_audit_signext.json"))["pools"])   # the sign extension's pools
neg = lambda n: any(v.get("negative", 0) > 0 for v in au.get(n, {}).values() if isinstance(v, dict))
loops = sorted(loops, key=lambda p: not neg(p["name"]))
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
    # an axis whose pools are not generated yet (H7, stage 4) is skipped and listed, its read left for mode eval
    miss=$(python -c "
import glob, os, sys, yaml
print(sum(not glob.glob(os.path.join(os.environ['DATA_DIR'], f\"{p['name']}_*_val_*amplitudes.npy\"))
          for p in yaml.safe_load(open(sys.argv[1]))['processes']))" "recipes/catalog_v3_heldout_$A.yaml")
    if [ "$miss" != 0 ]; then echo "V3_AXIS_SKIPPED $A: $miss pools not generated"; continue; fi
    P=("$R/$A"/*.yaml)
    # smoke: two probes per axis; on H5 one that changes sign and one that does not, so both scorings are exercised
    if [ "$MODE" = smoke ]; then
        P=($(python -c "
import json, os, sys
au = json.load(open('analysis/catalog_v3/pool_audit.json'))['pools']
neg = lambda f: au[os.path.basename(f)[:-5]]['val'].get('negative', 0) > 0
fs = sys.argv[2:]
print(' '.join(([f for f in fs if neg(f)][:1] + [f for f in fs if not neg(f)][:1]) if sys.argv[1] == 'H5' else fs[:2]))" "$A" "${P[@]}"))
    fi
    # every axis on log max(|M|^2, floor) for a sign-changing pool (the one-loop pools of H5, and the twin's tau
    # one-loop pools of H3p); a positive pool's target is log|M|^2 either way
    SM=(--signed-magnitude)
    python tools/zero_shot_parent.py --parent "$OUT" --save-dir "$PRED/$A" "${SM[@]}" "${P[@]}" | sed -u "s/^/$A /"
    [ "${PIPESTATUS[0]}" = 0 ] || exit 1
    # H5's sign-changing pools: the sign head's own scoring too (magnitude and wrong-sign share, the signed A/B's metrics)
    if grep -qE "name: .*_(nlo|loop)$" "recipes/catalog_v3_heldout_$A.yaml"; then
        # the sign head's scoring on every H5 pool: signed_ab_score.py scores those the run resolves as sign-changing
        # (any role non-positive, its abslog transform) and lists the others as skipped
        python tools/signed_ab_score.py --run "$OUT" --arm head "${P[@]}" | tee "$PRED/${A}_signed.txt"
        [ "${PIPESTATUS[0]}" = 0 ] || exit 1
    fi
done
if [ "$MODE" = smoke ]; then   # the fine-tune path from a catalog_v3 short parent, both fine-tune optimizers
    for O in AdamW MuonAdamW; do
        python tools/catalog_ft.py --parent "$OUT" --optimizer "$O" --out "$OUT/ft" --probes v3_gg_attx --only d2 \
            --steps-cap 20 || exit 1
    done
fi
echo "V3_SHORT_DONE $ARM"
