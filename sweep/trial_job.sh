#!/bin/bash
#SBATCH --job-name=trial
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# One detached DyHPO trial, submitted by sweep/drive.py through `site submit` on any site.
# Everything the trial needs arrives as arguments (--sweep-config, --hp-idx, --t-steps,
# --hp k=v ...); nothing is read from or written to a shared search state. The result goes
# back to the driver as one RESULT_JSON line on stdout. `site submit --hdr=--mem=8G` overrides the
# #SBATCH values above per sweep (time, cpus); memory is the site's business, added at submit
# (drive.py passes --hdr --mem= only where the site takes it; Jean Zay rejects the flag).
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
echo "site=${CCORCH_SITE:-?} host=$(hostname) run=${CCORCH_RUN_ID:-?} args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/TRIAL_GPU /'
# TRIAL_TIMEV=1 (a probe): the host peak memory and wall from /usr/bin/time -v, in the log
if [ "${TRIAL_TIMEV:-}" = 1 ] && [ -x /usr/bin/time ]; then
    exec /usr/bin/time -v python sweep/run_trial.py "$@"
fi
exec python sweep/run_trial.py "$@"
