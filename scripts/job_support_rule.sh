#!/bin/bash
#SBATCH --job-name=support_rule
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
# CPU only (selective_rescale, D561): kit-score one no-rescale run without and with the rescale below x0
# (analysis/selective_rescale/support_rule.py SCORE; prints its SRKIT line). Its own job, not `site run --job`, for the
# memory: kit_predict on the 1.2M-event u u~ g kit pool was killed at the CPU block's 2G (rc=-9, 2026-11-25).
#     site submit ccin2p3 FA scripts/job_support_rule.sh --item selective_rescale -- '<lab>#<run dir under runs/>' <REBUILD_SET>
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
KIT_THREADS=1 python analysis/selective_rescale/support_rule.py remote "$1" "$2" && echo "RUN_DONE"
