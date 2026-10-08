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
c = yaml.safe_load(open(sys.argv[1]))
fp = c.get("fixed_params") or {}
skip = ("run_name", "exp_name", "run_dir", "fine_tune.pretrained_path")
# ${PROJECT_DIR}, ${DATA_DIR}, ... are expanded here, as siteconf.resolve does for a sweep: Hydra would read them as
# config interpolations and fail
print(" ".join(f"{k}={os.path.expandvars(str(v))}" for k, v in fp.items() if k not in skip and v is not None))
EOF
)
RUN=zz_smoke_$(basename "$CFG" .yaml)_${SLURM_JOB_ID:-$$}
echo "SMOKE $CFG steps=$STEPS run=$RUN"
# validate_frac=0: a config's validate_frac (a fraction of the run) would override validate_every_n_steps
python run.py $OVR training.iterations=$STEPS training.validate_frac=0 training.validate_every_n_steps=100 \
  evaluation.train_subsample=2000 use_mlflow=false run_name=$RUN exp_name=zz_smoke "$@"
echo "SMOKE_EXIT $?"
