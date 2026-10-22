#!/bin/bash
#SBATCH --job-name=het23_eval
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# D75: the per-event test evaluation of each het23 sweep's best trial (its sigma stage already ran in the trial), as
# job_het_delta.sh writes it for the Delta cells, so it compares with the MSE cell best-vs-best. No retraining.
#     site submit ccin2p3 FA scripts/job_het23_eval.sh -- runs/<sweep>/trial_XXXX [...]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
rc=0
for OUT in "$@"; do
    python tools/eval_delta.py --out-dir "$SCRATCH/zz_delta_het23" "${OUT%/}/sigma_stage" || rc=1
done
exit $rc
