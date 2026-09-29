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
exec python sweep/run_trial.py "$@"
