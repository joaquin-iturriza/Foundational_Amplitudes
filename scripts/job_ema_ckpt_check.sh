#!/bin/bash
#SBATCH --job-name=ema_ckpt_check
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
# EMA checkpoint handling (D63) on real runs' models: tests/test_ema_checkpoint.py --real.
#     site submit <site> FA scripts/job_ema_ckpt_check.sh -- <run_dir> [<run_dir> ...]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
python tests/test_ema_checkpoint.py
python tests/test_ema_checkpoint.py --real --out-dir "$SCRATCH/zz_ema_ckpt_$SLURM_JOB_ID" "$@"
