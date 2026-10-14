#!/bin/bash
#SBATCH --job-name=eval_delta_seeds
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# Invariance check of the Delta evaluation: the same runs under other random Lorentz frames (tools/eval_delta.py --seed).
#     site submit <site> FA scripts/job_eval_delta_seeds.sh -- <list file> <seed> [<seed> ...]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
L="$1"; shift
for s in "$@"; do python tools/eval_delta.py --out-dir "$SCRATCH/zz_delta_seeds" --seed "$s" --list "$L"; done
