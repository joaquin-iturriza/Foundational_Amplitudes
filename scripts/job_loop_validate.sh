#!/bin/bash
#SBATCH --job-name=loop_validate
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=loop_validate_%j.out
#SBATCH --error=loop_validate_%j.out
# Synthetic one-loop test (D44), CPU only: builds the pools of recipes/loop_synthetic.yaml (train/val/test, the 19
# synthetic trees are cached already) through prebuild_recipes.py, then compares the 19 synthetic one-loop pools with
# the real one-loop probes' pools (analysis/loops/loop_validate.py) into runs/loop_val/ (site fetch <run> --dir runs/loop_val).
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set -e
python tests/test_synthetic_loops.py
python prebuild_recipes.py recipes/loop_synthetic.yaml --workers "${SLURM_CPUS_PER_TASK:-8}"
python analysis/loops/loop_validate.py --out runs/loop_val/results_loop_val.json
