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
#   v4_dimensionless_muon, v4_dimensionless_muon_seed2, dimensionless_muon_best_seed2, dimensionless_muon_best   catalog_v4 against catalog_v3 with
#                          Muon at its search's best (catalog_coverage; runs/catalog_v4_short/<arm>, $SCRATCH/zz_v4_short_<arm>)
# Mode smoke also fine-tunes the smoke parent on one held-out probe, 20 steps, with AdamW and with Muon (tools/catalog_ft.py;
# catalog_v3 arms only).
#   site submit ccin2p3 FA scripts/job_catalog_v3_short.sh --hdr=--mem=48G -- <arm> [smoke|eval]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
ARM="${1:?baseline, baseline_sign_pairs, baseline_sign_pairs_bounded, baseline_sign_near, baseline_more_signed, baseline_seed2, baseline_more_signed_seed2, baseline_more_signed_seed2_steplog, baseline_more_signed_muon, dimensionless, dimensionless_shat, dimensionless_hybrid, dimensionless_blend050, dimensionless_blend025, dimensionless_context, dimensionless_twin, dimensionless_pp, dimensionless_muon, dimensionless_seed2, dimensionless_muon_seed2, muon_search_best_adamw, muon_search_best_muon, sign_search_best, dimensionless_muon_best_seed2, dimensionless_muon_best, v4_dimensionless_muon or v4_dimensionless_muon_seed2}"; MODE="${2:-}"
SRC=runs/dancat_het_pre/trial_0058
TRAIN="$PROJECT_DIR/recipes/catalog_v3_train.yaml"
# the sign head for the signed one-loop pools; the diagram encoder's edge labels for the BSM layer's particles
COMMON=(training.loss=MSE ++training.loss_aggregation=mean ++training.sign_head=true
        ++model.diagram_encoder_llm4lhc.bsm_labels=true)
DIM=(++data.target_physics_factor=true ++data.target_physics_dim_only=true ++data.target_physics_s0=1.0
     ++data.dimensionless_inputs=true ++data.target_physics_orders=true ++data.target_physics_dim_ref=qperp
     ++data.shared_standardization=true ++data.affine_multiplicity_std=false)
# the Muon search's best (sweep/sweep_config_ccin2p3_muon_test_search_muon.yaml, trial_1860; par:v3_muon, D155)
MUON_BEST=(training.optimizer=MuonAdamW ++training.lr=0.004997423 ++training.muon_weight_decay=0.02171863)
CAT=v3; CROSS=()   # the catalog whose held-out axes the arm reads; extra reads after them (XV3, XV4, PP: axis_recipe)
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
    # the Muon test's stage 2 (D146): the best trial of each optimizer's 8-trial search at this horizon
    # (sweep/sweep_config_ccin2p3_muon_test_search_{adamw,muon}.yaml), read zero-shot only (mode eval) from its run dir
    muon_search_best_adamw) OUT_FIXED=runs/muon_test_search_adamw/trial_0768 ;;
    muon_search_best_muon)  OUT_FIXED=runs/muon_test_search_muon/trial_1860; CROSS=(XV4 PP) ;;
    # the sign head search's best trial by its objective val_total (sweep/sweep_config_ccin2p3_sign_head_search.yaml,
    # D155: pair sign head width 16, hidden 16, unbounded, no near-zero weighting, lr 1.246e-2), read zero-shot only
    # (mode eval) against baseline_more_signed_muon; catalog_v3's own axes only
    sign_search_best)       OUT_FIXED=runs/sign_head_search/trial_0177 ;;
    # catalog_v4's short pretrainings against catalog_v3's (catalog_coverage; D130 the coverage, D155 Muon): the
    # dimensionless arm with MuonAdamW at the Muon search's best (trial_1860 above: lr 4.997423e-3, decoupled weight
    # decay 0.02171863, the rest the arm's) on recipes/catalog_v4_train.yaml (1153 pools) at catalog_v3's 4264 steps:
    # same steps and batch, so about 6 epochs of each pool's 1e4 events where catalog_v3 has 5. Seeds 42 and 43; the
    # catalog_v3 pair is dimensionless_muon_best (trial_1860 retrained, seed 42) and dimensionless_muon_best_seed2 (seed 43). Besides its own
    # axes each reads the other catalog's pools of the probes held out in both (XV3/XV4,
    # recipes/make_catalog_v3v4_shared.py) and the 13 TeV pp pools (PP, recipes/catalog_v3_pp_test.yaml), which neither trains on
    dimensionless_muon_best_seed2) NORM=("${DIM[@]}" "${MUON_BEST[@]}" seed=43); CROSS=(XV4 PP) ;;
    # trial_1860 retrained at seed 42 with the order features and the target's alpha_s power from the diagrams
    # (experiment._orders_from_diagrams): trial_1860 had the table's alpha_s power on zp_uux_Y1g and thdm_uux_h2h3g
    # (2 of 1397 pools), so both catalog_v3 seeds now carry what catalog_v4's do
    dimensionless_muon_best) NORM=("${DIM[@]}" "${MUON_BEST[@]}"); CROSS=(XV4 PP) ;;
    v4_dimensionless_muon)  CAT=v4; NORM=("${DIM[@]}" "${MUON_BEST[@]}"); CROSS=(XV3 PP)
                            TRAIN="$PROJECT_DIR/recipes/catalog_v4_train.yaml"; ITERS_FIXED=4264 ;;
    v4_dimensionless_muon_seed2) CAT=v4; NORM=("${DIM[@]}" "${MUON_BEST[@]}" seed=43); CROSS=(XV3 PP)
                            TRAIN="$PROJECT_DIR/recipes/catalog_v4_train.yaml"; ITERS_FIXED=4264 ;;
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
    # the pair sign head again with its pair channels bounded (tanh) before their sums and products; the run writes
    # its per-step gradient norm (train_steps.npz), so a blow-up's lead-up is read
    baseline_sign_pairs_bounded) NORM=(++data.shared_standardization=true ++data.affine_multiplicity_std=false
                                 ++model.sign_pair_dim=16 ++model.sign_pair_bound=true) ;;
    # the baseline trained on more sign-changing one-loop pools (recipes/catalog_v3_train_signext.yaml: catalog_v3_train
    # plus the 95 one-loop pools of the coloured 2->3 training trees whose train events take both signs,
    # recipes/make_catalog_v3_train_signext.py), at the baseline's 4264 steps (same compute)
    baseline_more_signed)  NORM=(++data.shared_standardization=true ++data.affine_multiplicity_std=false)
                           TRAIN="$PROJECT_DIR/recipes/catalog_v3_train_signext.yaml"; ITERS_FIXED=4264 ;;
    # second seeds of the baseline and of the more-signed-pools arm (seed 43: model init and batch order; the pools and
    # their order unchanged), to size the third sign arm's gain and cost against seed spread
    baseline_seed2)        NORM=(++data.shared_standardization=true ++data.affine_multiplicity_std=false seed=43) ;;
    baseline_more_signed_seed2) NORM=(++data.shared_standardization=true ++data.affine_multiplicity_std=false seed=43)
                           TRAIN="$PROJECT_DIR/recipes/catalog_v3_train_signext.yaml"; ITERS_FIXED=4264 ;;
    # the more-signed-pools arm's seed 43 again, unchanged, in its own run dir so it trains (the first run's dir holds a
    # best checkpoint), now with the per-step record train_steps.npz: its blow-up at steps 1360 to 1445 read step by step
    baseline_more_signed_seed2_steplog) NORM=(++data.shared_standardization=true ++data.affine_multiplicity_std=false seed=43)
                           TRAIN="$PROJECT_DIR/recipes/catalog_v3_train_signext.yaml"; ITERS_FIXED=4264 ;;
    # the sign head search's comparison point (sweep/sweep_config_ccin2p3_sign_head_search.yaml, D155): the
    # more-signed-pools arm with MuonAdamW at the Muon search's best (lr 5e-3, decoupled weight decay 0.0217,
    # par:v3_muon), no pair head, no near-zero weighting; its result json carries val_total, the search's objective
    baseline_more_signed_muon) NORM=(++data.shared_standardization=true ++data.affine_multiplicity_std=false
                           ++training.optimizer=MuonAdamW ++training.lr=0.005 ++training.muon_weight_decay=0.0217)
                           TRAIN="$PROJECT_DIR/recipes/catalog_v3_train_signext.yaml"; ITERS_FIXED=4264 ;;
    baseline_sign_near)    NORM=(++data.shared_standardization=true ++data.affine_multiplicity_std=false
                                 ++training.sign_near_weight=10 ++training.sign_near_factor=10) ;;
    *) echo "unknown arm $ARM"; exit 1 ;;
esac
OUT="$PROJECT_DIR/${OUT_FIXED:-runs/catalog_${CAT}_short/$ARM}"; PRED="$SCRATCH/zz_${CAT}_short_$ARM"
# an axis's recipe: the arm's catalog's held-out axis, or one of the cross reads
axis_recipe() { case "$1" in
    XV3) echo recipes/catalog_v3_shared_heldout.yaml ;; XV4) echo recipes/catalog_v4_shared_heldout.yaml ;;
    PP) echo recipes/catalog_v3_pp_test.yaml ;; *) echo "recipes/catalog_${CAT}_heldout_$1.yaml" ;; esac; }
[ -n "$OUT_FIXED" ] && [ "$MODE" != eval ] && { echo "$ARM is a sweep trial: mode eval only"; exit 1; }
# 5 epochs of 1e4 events per process at batch 16384, from the recipe in use (1397 pools: 4264 steps; the twin's 1246: 3803)
ITERS=${ITERS_FIXED:-$(python -c "import math, sys, yaml; print(math.ceil(len(yaml.safe_load(open(sys.argv[1]))['processes']) * 1e4 / 16384 * 5))" "$TRAIN")}
EXTRA=(training.validate_frac=0.02 training.iterations=$ITERS ++data.memmap=false ++data.train_subsample=10000
       ++data.eval_subsample=200 evaluation.train_subsample=2000 data.processes_file="$TRAIN")
AXES=(H1 H2 H2p5 H3 H5 H6 H7E H7c "${CROSS[@]}"); [ "$ARM" = dimensionless_twin ] && AXES+=(H3p H4)
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
    AXES=(H1 H5 "${CROSS[@]}")
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
# the held-out recipes this job reads: a folder of its own, made fresh (plain mkdir refuses an existing one). The folder
# was once one per axis shared by every job of both catalogs and never emptied, so catalog_v3 runs read catalog_v4's
# probes and the reverse (baseline_more_signed_muon: 62 on H1, 28 on H5); jobs rewriting it while another read it
# could also lose a read
R="$SCRATCH/zz_${CAT}_heldout_recipes_${ARM}_${SLURM_JOB_ID:-$$}"
mkdir "$R" || { echo "V3_RECIPES $R exists: refusing to read from a folder this job did not make"; exit 1; }
for A in "${AXES[@]}"; do
    AR=$(axis_recipe "$A")   # one recipe per process, keeping the axis file's top-level keys (sampling: mixture)
    mkdir "$R/$A" || exit 1
    # an earlier job's predictions for this arm and axis are moved aside (never deleted): the folder then holds this
    # job's reads only, so a probe this job fails cannot be answered by a stale prediction
    if [ -d "$PRED/$A" ] && [ -n "$(ls -A "$PRED/$A")" ]; then mv "$PRED/$A" "$PRED/$A.prev_${SLURM_JOB_ID:-$$}" || exit 1; fi
    mkdir -p "$PRED/$A"
    python - "$R/$A" "$AR" <<'PY'
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
          for p in yaml.safe_load(open(sys.argv[1]))['processes']))" "$AR")
    if [ "$miss" != 0 ]; then echo "V3_AXIS_SKIPPED $A: $miss pools not generated"; continue; fi
    P=("$R/$A"/*.yaml)
    # smoke: two probes per axis; on H5 one that changes sign and one that does not, so both scorings are exercised
    if [ "$MODE" = smoke ]; then
        P=($(python -c "
import json, os, sys
au = json.load(open('analysis/catalog_v3/pool_audit.json'))['pools']
neg = lambda f: au.get(os.path.basename(f)[:-5], {}).get('val', {}).get('negative', 0) > 0   # catalog_v3's audit
fs = sys.argv[2:]
print(' '.join(([f for f in fs if neg(f)][:1] + [f for f in fs if not neg(f)][:1]) if sys.argv[1] == 'H5' else fs[:2]))" "$A" "${P[@]}"))
    fi
    # every recipe about to be read is one of this axis's recipe and of the run's catalog (its probes' name prefix: the
    # arm's catalog on its own axes, v3_ / v4_ / pp13_ on the cross reads); outside smoke, all of the axis's probes
    python - "$CAT" "$A" "$AR" "$MODE" "${P[@]}" <<'PY' || exit 1
import os, sys, yaml
cat, axis, ar, mode, files = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
pfx = {"XV3": "v3_", "XV4": "v4_", "PP": "pp13_"}.get(axis, f"{cat}_")
names = {p["name"] for p in yaml.safe_load(open(ar))["processes"]}
got = [os.path.basename(f)[:-len(".yaml")] for f in files]
bad = sorted(n for n in got if n not in names or not n.startswith(pfx))
if bad:
    sys.exit(f"V3_RECIPE_CHECK FAIL {axis}: {len(bad)} recipes not in {os.path.basename(ar)} or not {pfx}*: {bad[:10]}")
if mode != "smoke" and set(got) != names:
    sys.exit(f"V3_RECIPE_CHECK FAIL {axis}: {len(names - set(got))} probes of {os.path.basename(ar)} without a recipe")
print(f"V3_RECIPE_CHECK {axis}: {len(got)} recipes, all {pfx}* probes of {os.path.basename(ar)}")
PY
    # every axis on log max(|M|^2, floor) for a sign-changing pool (the one-loop pools of H5, and the twin's tau
    # one-loop pools of H3p); a positive pool's target is log|M|^2 either way
    SM=(--signed-magnitude)
    python tools/zero_shot_parent.py --parent "$OUT" --save-dir "$PRED/$A" "${SM[@]}" "${P[@]}" | sed -u "s/^/$A /"
    [ "${PIPESTATUS[0]}" = 0 ] || exit 1
    # H5's sign-changing pools: the sign head's own scoring too (magnitude and wrong-sign share, the signed A/B's metrics)
    if grep -qE "name: .*_(nlo|loop)$" "$AR"; then
        # the sign head's scoring on every H5 pool: signed_ab_score.py scores those the run resolves as sign-changing
        # (any role non-positive, its abslog transform) and lists the others as skipped
        python tools/signed_ab_score.py --run "$OUT" --arm head "${P[@]}" | tee "$PRED/${A}_signed.txt"
        [ "${PIPESTATUS[0]}" = 0 ] || exit 1
    fi
done
if [ "$MODE" = smoke ] && [ "$CAT" = v3 ]; then   # the fine-tune path from a catalog_v3 short parent, both fine-tune optimizers
    for O in AdamW MuonAdamW; do
        python tools/catalog_ft.py --parent "$OUT" --optimizer "$O" --out "$OUT/ft" --probes v3_gg_attx --only d2 \
            --steps-cap 20 || exit 1
    done
fi
rm -rf "$R"   # this job's own recipe folder, made above
echo "V3_SHORT_DONE $ARM"
