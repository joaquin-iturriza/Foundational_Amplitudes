#!/bin/bash
#SBATCH --job-name=memmap_ab
#SBATCH --cpus-per-task=5
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --output=memmap_ab_%j.out
#SBATCH --error=memmap_ab_%j.out
#
# data.memmap equivalence on the GPU: a few training steps with the flag off, on and off again
# (tests/test_memmap_data.py section 3), on the first processes of a sweep's recipe.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
python tests/test_memmap_data.py "$@"
