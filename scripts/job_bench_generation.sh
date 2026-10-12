#!/bin/bash
#SBATCH --job-name=bench_generation
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=bench_generation_%j.out
#SBATCH --error=bench_generation_%j.out
# One-core generation rates through build_mixture_dataset (tools/bench_synthetic.py), CPU only (no GPU): the
# MadGraph high-multiplicity processes next to the synthetic versions, on the same node. Arguments come in
# BENCH_ARGS, words joined by '+' (site submit passes none), e.g.
#   site submit ccin2p3 FA scripts/job_bench_generation.sh --item <item> \
#       --env BENCH_ARGS=--real+uubar_Zgggg+uubar_Zggggg+udbar_Wgggg+udbar_Wggggg+--n+2000
# A missing C++ standalone is generated and compiled first (datagen.ensure_backend; the 4 cores are for that),
# timed apart from the rate; the rate itself is single-threaded.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ -n "${BENCH_ARGS:-}" ]; then set -- ${BENCH_ARGS//+/ }; fi
python tools/bench_synthetic.py "$@"
