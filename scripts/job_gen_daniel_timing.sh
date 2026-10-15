#!/bin/bash
#SBATCH --job-name=gen_daniel_timing
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --output=gen_daniel_timing_%j.out
#SBATCH --error=gen_daniel_timing_%j.out
# CPU only (no GPU). Times the catalog_daniel lhe_pp path (arXiv:2606.23791 regenerated, D51/D54) for
# single processes before the full run (scripts/job_gen_daniel_catalog.sh): MG5 output + standalone
# compile, MadEvent generate_events (their cards, one core), LHE parsing, labelling, at 10k and 100k
# events. Default processes: dc_gb_bbbx (2->3, massive initial b) and dc_gg_ggg (2->3 with the most
# diagrams of the catalog, 25, and a low unweighting efficiency: the slow end). Everything is written
# to a temporary directory under $SCRATCH and deleted; no pool cache is touched. Arguments come in
# TIMING_ARGS, words joined by '+' (the submitter passes no positional arguments), e.g.
#   --env TIMING_ARGS=--process+dc_uux_emepg+--n+10000
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ -n "${TIMING_ARGS:-}" ]; then
  set -- ${TIMING_ARGS//+/ }
else
  set -- --process dc_gb_bbbx --process dc_gg_ggg --n 10000 --n 100000
fi
echo "host $(hostname)  cpu$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)  MG5 $(sed -n 's/^version = //p' "${MG5_WORK_DIR:-$WORK/mg5amcnlo}/VERSION" 2>/dev/null)"
python tools/time_lhe_pp.py "$@"
