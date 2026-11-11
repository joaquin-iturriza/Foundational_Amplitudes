#!/bin/bash
#SBATCH --job-name=kit_pkg
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# kit_benchmark step 5: build the KIT checkpoint + inference package (tools/kit_package.py build) and check that it
# alone reproduces every packaged run's predictions (kit_package.py check, CPU, float64). CPU only.
#     site submit ccin2p3 FA scripts/job_kit_package.sh --item kit_benchmark -- <out dir> <run_dir>[=<label>] ...
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
OUT="$1"; shift
python tools/kit_package.py build --out "$OUT" --runs "$@" && python tools/kit_package.py check --pkg "$OUT" \
    && echo "RUN_DONE"
