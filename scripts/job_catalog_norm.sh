#!/bin/bash
#SBATCH --job-name=catalog_norm
#SBATCH --cpus-per-task=5
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=catalog_norm_%j.out
#SBATCH --error=catalog_norm_%j.out
#
# D83 (the user: "send the A against B"): the catalog pretraining at hp58's configuration and HPs
# (runs/dancat_het_pre/trial_0058; CLAUDE.md A/B step 3) under a normalization that needs no data of a process, then
# the 139 held-out processes read with it (tools/zero_shot_parent.py, per-event predictions in $SCRATCH/zz_catalog_norm_<arm>/):
#   B  one standardization shared by all processes (data.shared_standardization)
#   A  the same after each event's |M|^2 is multiplied by (s0/s_hat)^(4-N)/C_p (data.target_physics_factor; s0 =
#      (17 GeV)^2, the reference fig:norm_factor chose on the training processes)
# 50 validations (D79). smoke: 19 catalog processes, 40 steps, 2 held-out processes (a test of the code path only).
#   site submit ccin2p3 FA scripts/job_catalog_norm.sh --hdr=--mem=40G -- A|B [smoke|eval]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
ARM="${1:?A, Adim, B, Bgev, Btev, Dless, DlessIn, DlessInVirt, DlessInNoDiag, DlessInOrd, Dpair or Ds0}"; SMOKE="${2:-}"
case "$ARM" in A|Adim|B|Bgev|Btev|Dless|DlessIn|DlessInVirt|DlessInNoDiag|DlessInOrd|Dpair|Ds0) ;; *) echo "unknown arm $ARM"; exit 1 ;; esac
# the probes the run is scored on: the catalog's 139 held-out (default) or 2to4, the 40 2->4 processes beyond the catalog
# (recipes/gen_catalog_2to4.py, D90), predictions in <PRED>_2to4
SET="${3:-catalog}"
SRC=runs/dancat_het_pre/trial_0058
NORM=(++data.shared_standardization=true ++data.affine_multiplicity_std=false)
[ "$ARM" = A ] && NORM+=(++data.target_physics_factor=true ++data.target_physics_s0=289.0)
# Adim (2026-10-22): the mass-dimension factor (s0/s_hat)^(4-N) alone, no coupling product
[ "$ARM" = Adim ] && NORM+=(++data.target_physics_factor=true ++data.target_physics_s0=289.0 ++data.target_physics_dim_only=true)
# the extrapolation round (the user, 2026-10-22): T = log|M|^2 - (4-N) log(Lambda^2), one shared standardization on top
#   Bgev  Lambda = 1 GeV (B as run); Btev  Lambda = 1 TeV (a pure change of units, 1e6 GeV^2);
#   Dless Lambda^2 = s_hat per event, s0 = 1 (|M|^2 s_hat^(N-4), dimensionless); Ds0 the same with s0 = 289 GeV^2 (Adim)
DIM=(++data.target_physics_factor=true ++data.target_physics_dim_only=true)
[ "$ARM" = Btev ]  && NORM+=("${DIM[@]}" ++data.target_physics_dim_ref=1000000.0 ++data.target_physics_s0=1.0)
[ "$ARM" = Dless ] && NORM+=("${DIM[@]}" ++data.target_physics_s0=1.0)
[ "$ARM" = Ds0 ]   && NORM+=("${DIM[@]}" ++data.target_physics_s0=289.0)
# DlessIn (the user, 2026-10-22, the dimensionless path): Dless's target with dimensionless inputs too
# (data.dimensionless_inputs: momenta per event / sqrt(s_hat), masses as log10(m / sqrt(s_hat)), off-shellness / s_hat);
# in memory, the only path that implements it
# Dpair (2026-10-22): the event's own scale, Lambda^2 = the geometric mean of the pair invariants 2 p_i.p_j, s0 = 1
[ "$ARM" = Dpair ] && NORM+=("${DIM[@]}" ++data.target_physics_dim_ref=pairs ++data.target_physics_s0=1.0)
[ "$ARM" = DlessIn ] && NORM+=("${DIM[@]}" ++data.target_physics_s0=1.0 ++data.dimensionless_inputs=true ++data.memmap=false)
# the diagram comparison (D93, the user 2026-10-23: "compare also what daniel's diagram encoder is able to do by itself"):
# DlessIn's target and inputs, with the arXiv:2606.23791 encoder (hp58's, topology only: DlessIn itself), with its pooling
# weighted per event by each diagram's propagator virtualities (DlessInVirt, model.diagram_encoder_llm4lhc.virtuality_pool,
# "option 1 looks good for now"), and without diagrams (DlessInNoDiag, the floor)
[ "$ARM" = DlessInVirt ] && NORM+=("${DIM[@]}" ++data.target_physics_s0=1.0 ++data.dimensionless_inputs=true ++data.memmap=false
                                   ++model.diagram_encoder_llm4lhc.virtuality_pool=true)
[ "$ARM" = DlessInNoDiag ] && NORM+=("${DIM[@]}" ++data.target_physics_s0=1.0 ++data.dimensionless_inputs=true ++data.memmap=false
                                     ++model.use_diagrams=false)
# DlessInOrd (D93, the coupling-order half of the target): DlessIn with each event's |M|^2 also divided by
# (4 pi alpha_s_event)^q (4 pi alpha)^e, q and e the process's orders (data.target_physics_orders; per-event alpha_s from
# <pool>.alphas.npy, not needed where q = 0)
[ "$ARM" = DlessInOrd ] && NORM+=("${DIM[@]}" ++data.target_physics_s0=1.0 ++data.dimensionless_inputs=true ++data.memmap=false
                                  ++data.target_physics_orders=true)
EXTRA=(training.validate_frac=0.02)
# the memory-mapped cache on the node's local disk: /sps/lpnhe was 98% full (2026-10-22, the first A run failed building
# it there); built by this job, removed when it exits
MMAP="${TMPDIR:-/tmp}/fa_mmap_${SLURM_JOB_ID}"; mkdir -p "$MMAP"; trap 'rm -rf "$MMAP"' EXIT
df -h "$MMAP" | tail -1
NORM+=(++data.memmap_dir="$MMAP")
OUT="$PROJECT_DIR/runs/catalog_norm/hp58_$ARM"; PRED="$SCRATCH/zz_catalog_norm_$ARM"
# the probe set: the catalog's 139 held-out; 2to4, the 40 2->4 processes beyond the catalog; 2to4c30, the same with
# ptj = ptb = pta = 30 GeV (gen_catalog_2to4.py --cut 30, the cut test); 2to4E100, their 40 2->3 training parents at
# pp 100 TeV (--parents-at 100, the energy test); 2to4E100p, the 2->4 probes themselves at 100 TeV (--probes-at
# 100, energy and multiplicity together). Predictions in <PRED>_<SET> for every set but the catalog
case "$SET" in
    catalog)  R="$SCRATCH/zz_catalog_heldout_recipes"; HELD=recipes/catalog_daniel_heldout.yaml; PFX=dc_ ;;
    # DlessInOrd needs alpha_s on every probe with q > 0: d4_gdx_dxzh_g's could not be recovered, so its regeneration
    # d4r_gdx_dxzh_g stands in (catalog_2to4r_processes.yaml), from a recipe dir of its own
    2to4)     R="$SCRATCH/zz_catalog_2to4_recipes"; HELD=recipes/catalog_2to4_heldout.yaml; PFX=d4_
              [ "$ARM" = DlessInOrd ] && { R="$SCRATCH/zz_catalog_2to4_orders_recipes"; HELD=recipes/catalog_2to4_heldout_orders.yaml; PFX=d4; } ;;
    loop)     R="$SCRATCH/zz_catalog_loop_recipes"; HELD=recipes/catalog_loop_probes.yaml; PFX=""; WINDOWS=1 ;;
    top)      R="$SCRATCH/zz_catalog_top_recipes"; HELD=recipes/catalog_daniel_top.yaml; PFX=dc_ ;;
    2to4c30)  R="$SCRATCH/zz_catalog_2to4c30_recipes"; HELD=recipes/catalog_2to4c30_heldout.yaml; PFX=d4c30_
              [ "$ARM" = DlessInOrd ] && { R="$SCRATCH/zz_catalog_2to4c30_orders_recipes"; HELD=recipes/catalog_2to4c30_heldout_orders.yaml; PFX=d4c30; } ;;
    2to4E100) R="$SCRATCH/zz_catalog_2to4E100_recipes"; HELD=recipes/catalog_2to4E100_heldout.yaml; PFX=dE100_ ;;
    2to4E100p) R="$SCRATCH/zz_catalog_2to4E100p_recipes"; HELD=recipes/catalog_2to4E100p_heldout.yaml; PFX=d4E100_ ;;
    *) echo "unknown probe set $SET"; exit 1 ;;
esac
mkdir -p "$R"
if [ "$SMOKE" = smoke ] || [ "$SMOKE" = smoke_eval ]; then
    OUT="$SCRATCH/zz_catalog_norm_smoke_${ARM}_$SLURM_JOB_ID"; PRED="$OUT/preds"
    SUB="$OUT.recipe.yaml"
    python - "$SUB" <<'PY'
import random, sys, yaml
r = yaml.safe_load(open("recipes/catalog_daniel_train.yaml"))
random.Random(0).shuffle(r["processes"]); r["processes"] = r["processes"][:19]
yaml.safe_dump(r, open(sys.argv[1], "w"))
PY
    EXTRA=(training.iterations=40 training.validate_frac=0.5 evaluate=false data.processes_file="$SUB"
           ++training.heterosc_sigma_stage_steps=10)
    # smoke_eval: the post-training evaluation on (train subsampled), so the sigma stage writes its preds_*.npz
    [ "$SMOKE" = smoke_eval ] && EXTRA=("${EXTRA[@]/evaluate=false/evaluate=true}" evaluation.train_subsample=2000)
fi
# short (the user, 2026-10-22: "reduce the training set to 10000 events per process", "only 5 epochs", "keep events in
# memory", "Reduce the validation pool proportionally too", a shorter sigma stage): 1e4 train and 200 val/test events
# per process (hp58: 1e5 and 2000), 5 epochs (1352 x 1e4 / 16384 x 5 = 4126 steps), in memory, 50 validations, a 150-step sigma stage with
# 5 validations; runs/catalog_norm_short/<arm>, held-out predictions $SCRATCH/zz_catalog_norm_short_<arm> (scored on
# 2000 val events per held-out process, tools/zero_shot_parent.py --eval-subsample)
if [ "$SMOKE" = short ]; then
    OUT="$PROJECT_DIR/runs/catalog_norm_short/$ARM"; PRED="$SCRATCH/zz_catalog_norm_short_$ARM"
    EXTRA=(training.validate_frac=0.02 training.iterations=4126 ++data.memmap=false ++data.train_subsample=10000
           ++data.eval_subsample=200 ++training.heterosc_sigma_stage_steps=150
           ++training.heterosc_sigma_stage_validations=5 evaluation.train_subsample=2000)
fi
# extrap (the user, 2026-10-22, "send those jobs now too"): train on the 236 2->2 training processes only, all 1e5 of
# their train events (1e4 would give 720 steps over 5 epochs), the short runs' 4126 steps and settings otherwise;
# read on the 139 held-out, the 2->3 ones being the multiplicity extrapolation; runs/catalog_norm_extrap/<arm>. With 2->2
# only in training (4-N = 0 on every training event), Btev trains on Bgev's targets and Ds0 on Dless's: each pair is one
# network, read back differently at N = 5 (by (4-N) log 1e6 and (4-N) log 289), so only Bgev and Dless are trained
if [ "$SMOKE" = extrap ]; then
    OUT="$PROJECT_DIR/runs/catalog_norm_extrap/$ARM"; PRED="$SCRATCH/zz_catalog_norm_extrap_$ARM"
    SUB="$SCRATCH/zz_catalog_train_2to2_${SLURM_JOB_ID}.yaml"
    python - "$SUB" <<'PY'
import json, sys, yaml
m = {"dc_" + p["folder"]: p for p in json.load(open("analysis/catalog_daniel/classes_default.json"))["processes"]}
r = yaml.safe_load(open("recipes/catalog_daniel_train.yaml"))
r["processes"] = [p for p in r["processes"] if len(m[p["name"]]["process"].replace(">", " ").split()) == 4]
yaml.safe_dump(r, open(sys.argv[1], "w")); print(len(r["processes"]), "2->2 training processes")
PY
    EXTRA=(training.validate_frac=0.02 training.iterations=4126 ++data.memmap=false data.processes_file="$SUB"
           ++data.eval_subsample=200 ++training.heterosc_sigma_stage_steps=150
           ++training.heterosc_sigma_stage_validations=5 evaluation.train_subsample=2000)
fi
# eval: the held-out scoring only, from the run's best checkpoint (a run that blew up is reported at its best
# checkpoint, CLAUDE.md; B diverged at step 8734 on 2026-10-22). zero_shot_parent.py reads models/model_run0_best.pt.
# eval_short / eval_extrap: the held-out scoring of a finished short or extrap run, from its sigma stage
[ "$SMOKE" = eval_short ] && { OUT="$PROJECT_DIR/runs/catalog_norm_short/$ARM"; PRED="$SCRATCH/zz_catalog_norm_short_$ARM"; SMOKE=eval; }
[ "$SMOKE" = eval_extrap ] && { OUT="$PROJECT_DIR/runs/catalog_norm_extrap/$ARM"; PRED="$SCRATCH/zz_catalog_norm_extrap_$ARM"; SMOKE=eval; }
# FA_HOLDOUT=top: the particle-content test (D93, "every process with ... a top held out of training"): trained without the
# catalog's 142 processes with a top (recipes/catalog_daniel_train_notop.yaml, 1210 processes; the same steps, HPs and
# settings), into <OUT>_notop and <PRED>_notop; scored on SET=top (all 162 catalog processes with a top)
if [ -n "${FA_HOLDOUT:-}" ] && [ "$SMOKE" != short ] && [ "$SMOKE" != eval ]; then
    echo "FA_HOLDOUT is for the short runs and their scoring (mode $SMOKE sets its own processes_file)"; exit 1; fi
if [ "${FA_HOLDOUT:-}" = top ]; then
    OUT="${OUT}_notop"; PRED="${PRED}_notop"; EXTRA+=(data.processes_file="$PROJECT_DIR/recipes/catalog_daniel_train_notop.yaml")
elif [ -n "${FA_HOLDOUT:-}" ]; then echo "unknown FA_HOLDOUT $FA_HOLDOUT"; exit 1; fi
# FA_SEED: a rerun at another seed (a run that diverged, 2026-10-23 Dpair), into <OUT>_seed<N> and <PRED>_seed<N>
if [ -n "${FA_SEED:-}" ]; then
    OUT="${OUT}_seed$FA_SEED"; PRED="${PRED}_seed$FA_SEED"; EXTRA+=(seed=$FA_SEED)
fi
if [ "$SMOKE" = eval ]; then
    ls "$OUT"/models/model_run0_best.pt* >/dev/null 2>&1 || { echo "no best checkpoint in $OUT"; exit 1; }
elif [ ! -e "$OUT/sigma_stage/models" ]; then
    test -e "$OUT" && mv "$OUT" "$OUT.partial_$SLURM_JOB_ID"
    python run.py --config-path "$PROJECT_DIR/$SRC" --config-name config run_dir="$OUT" run_name=null warm_start_idx=null \
        base_dir="$PROJECT_DIR" training.is_dyhpo_run=false training.increment_steps=null \
        training.result_path="$OUT/result.json" plot=true "${NORM[@]}" "${EXTRA[@]}" || exit 1
fi
grep -a -m1 "target_physics_factor ON" "$OUT"/out_*.log 2>/dev/null
python - "$R" "$HELD" "$SET" <<'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[2]))
for p in r["processes"]:
    # the loop set holds several sqrt(s) windows of some processes: one recipe file per window
    stem = f"{p['name']}_{p['sqrts'][0]}-{p['sqrts'][1]}GeV" if sys.argv[3] == "loop" else p["name"]
    yaml.safe_dump({"processes": [p]}, open(f"{sys.argv[1]}/{stem}.yaml", "w"))
print(len(r["processes"]), "held-out recipes")
PY
P=("$R"/${PFX}*.yaml); [[ "$SMOKE" = smoke* ]] && [ "$SET" = catalog ] && P=("$R/dc_scx_wmhh.yaml" "$R/dc_ssx_vevex.yaml")
# FA_ONLY: score only these processes (space-separated names) into a zz_debug dir, e.g. with CUDA_LAUNCH_BLOCKING=1 to
# locate a crash (2026-10-23: DlessInVirt, an illegal memory access on dE100_ddx_hvmvmx, twice)
if [ -n "${FA_ONLY:-}" ]; then P=(); for n in $FA_ONLY; do P+=("$R/$n.yaml"); done; DEBUG_PRED=1; fi
# eval: the sigma-stage model when the run has one (trunk and mu frozen, so mu is the run's; its sigma is the calibrated
# one, D70), into its own prediction dir
# the sigma stage's model whenever the run has one (mu is the run's, frozen; sigma is the calibrated one): the held-out
# pulls are then the calibrated sigma's (before 2026-10-22 only eval mode did this; the short and extrap runs' first
# scoring read sigma before the stage)
PAR="$OUT"
if ls "$OUT"/sigma_stage/models/model_run0_best.pt* >/dev/null 2>&1; then PAR="$OUT/sigma_stage"; PRED="${PRED}_sigstage"; fi
[ "$SET" != catalog ] && PRED="${PRED}_$SET"
[ -n "${DEBUG_PRED:-}" ] && PRED="$SCRATCH/zz_debug_${ARM}_${SET}_$SLURM_JOB_ID"
mkdir -p "$PRED"
python tools/zero_shot_parent.py --parent "$PAR" --save-dir "$PRED" ${WINDOWS:+--window-names} "${P[@]}"
