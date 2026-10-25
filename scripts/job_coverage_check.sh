#!/bin/bash
#SBATCH --job-name=cov_check
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=cov_check_%j.out
#SBATCH --error=cov_check_%j.out
#
# CPU only (D127): the catalog's coverage check (analysis/catalog_v3/coverage_check.py). Every catalog_v3 tree pool
# sampled under the current cuts, the proposed analysis-level coverage with the forward component, and the same on a
# wide log-uniform sqrt(s_hat) window; labelled by the compiled backends, read against the current pools and the 13 TeV
# pp pools; the one-loop and loop-induced pools read without new labels (the script's docstring says why). Appends rows
# to $SCRATCH/zz_cov_check/rows_v2.jsonl (resumable: a rerun skips the entries already there).
# No pool is regenerated and nothing is written to $DATA_DIR.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
python analysis/catalog_v3/coverage_check.py run --n "${COV_N:-2000}" --workers "${SLURM_CPUS_PER_TASK:-32}"
