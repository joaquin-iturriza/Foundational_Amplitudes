#!/bin/bash
#SBATCH --job-name=fa_smoke
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=fa_smoke_%j.out
#SBATCH --error=fa_smoke_%j.err
#
# A short GPU run of a sweep config's fixed setup, to check a new code path before its real run (CLAUDE.md: never
# test infrastructure with a real training job). It uses the config's fixed_params at the default lr for STEPS steps,
# with plots on and a small train-split evaluation:
#   site submit <site> FA scripts/job_config_smoke.sh -- sweep/sweep_config_X.yaml [STEPS] [extra overrides...]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
CFG=$1; STEPS=${2:-300}; shift 2 2>/dev/null || shift $#
OVR=$(python - "$CFG" <<'EOF'
import os, sys, yaml
sys.path.insert(0, "sweep")
c = yaml.safe_load(open(sys.argv[1]))
fp = c.get("fixed_params") or {}
base = os.path.expandvars(str(c["base_config"])) if c.get("base_config") else None
# with a base_config (a fine-tune search: the parent run's config.yaml), as run_trial.py builds a trial: that config,
# its other-site paths re-rooted, every override ++, the parent checkpoint kept
skip = ("run_name", "exp_name", "run_dir") + (() if base else ("fine_tune.pretrained_path",))
out = []
if base:
    from run_trial import _rerooted_base_paths
    out = ["--config-path", base, "--config-name", "config"] + _rerooted_base_paths(base)
pre = "++" if base else ""
# ${PROJECT_DIR}, ${DATA_DIR}, ... are expanded here, as siteconf.resolve does for a sweep: Hydra would read them as
# config interpolations and fail
out += [f"{pre}{k}={os.path.expandvars(str(v))}" for k, v in fp.items() if k not in skip and v is not None]
print(" ".join(out))
EOF
)
# unique per submission: on HTCondor there is no SLURM_JOB_ID and the PID repeats (2026-10-08: two runs both "85")
RUN=zz_smoke_$(basename "$CFG" .yaml)_${SLURM_JOB_ID:-$(date +%s)_$RANDOM}
echo "SMOKE $CFG steps=$STEPS run=$RUN"
# validate_frac=0: a config's validate_frac (a fraction of the run) would override validate_every_n_steps.
# sigma stage: 2 validations, not 25 (each is a full validation pass: 25 took 2.5 h at 64 heads, FA-ccin2p3-1cdc28b2)
python run.py $OVR training.iterations=$STEPS training.validate_frac=0 training.validate_every_n_steps=100 \
  training.heterosc_sigma_stage_validations=2 evaluation.train_subsample=2000 use_mlflow=false run_name=$RUN exp_name=zz_smoke \
  run_dir="$PROJECT_DIR/runs/zz_smoke/$RUN" "$@"   # explicit: a base_config's run_dir is the parent run's own
RC=$?
echo "SMOKE_EXIT $RC"
[ "$RC" = 0 ] && echo "RUN_DONE $PROJECT_DIR/runs/zz_smoke/$RUN"   # the supervisor's end marker
exit $RC
