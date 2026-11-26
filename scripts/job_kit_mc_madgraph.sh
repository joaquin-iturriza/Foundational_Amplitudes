#!/bin/bash
#SBATCH --job-name=kit_mcmg
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# kit_benchmark: Bakar's slide-11 MC test with MadGraph (tools/kit_mc_madgraph.py), CPU only: MadEvent events of one
# process at the KIT settings, their exact |M|^2 (checked against KIT's), the surrogate on them, the metrics.
#     site submit <site> FA scripts/job_kit_mc_madgraph.sh --item kit_benchmark -- <ee_uug|ee_uugg> <dir> <N> <run_dir>
#     ... -- tt <run_dir>        # the one-loop virtual: the threshold scan's integrated-correction bias (kit_threshold_scan)
#     ... -- tt_bulk <run_dir>   # the same over KIT's whole tt bulk window 353-1739 GeV (kit_threshold_scan --window bulk)
#     ... -- surr <ee_uug|ee_uugg> <dir> <N> <run_dir>   # slide 11 as written: MadEvent on the surrogate's |M_hat|^2
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
export KIT_THREADS=1     # every surrogate read at 1 thread: the package's predictor and the served MadEvent integrand
if [ "$1" = tt ]; then
    D=runs/kit/threshold_scan
    python tools/kit_threshold_scan.py predict --dir "$D" "$2" && \
    python tools/kit_threshold_scan.py collect --dir "$D" --out "$D/kit_threshold_scan_mcmg.json" && echo "RUN_DONE"; exit
fi
if [ "$1" = tt_bulk ]; then
    D=runs/kit/tt_bulk_window
    python tools/kit_threshold_scan.py build --window bulk --dir "$D" && \
    python tools/kit_threshold_scan.py predict --dir "$D" "$2" && \
    python tools/kit_threshold_scan.py collect --dir "$D" --out "$D/kit_tt_bulk_window.json" && echo "RUN_DONE"; exit
fi
if [ "$1" = surr ]; then
    P="$2"; D="$3"; N="$4"; RUN="$5"
    python tools/kit_mc_madgraph.py gen "$P" --dir "$D" --n "$N" --surrogate "$RUN" && \
    python tools/kit_mc_madgraph.py predict "$P" --dir "$D" "$RUN" --surr && \
    python tools/kit_mc_madgraph.py predict "$P" --dir "$D" "$RUN" && \
    python tools/kit_mc_madgraph.py metrics --dir "$D" --out "$D/kit_mc_madgraph_${P}_surr.json" && echo "RUN_DONE"; exit
fi
P="$1"; D="$2"; N="$3"; RUN="$4"
python tools/kit_mc_madgraph.py gen "$P" --dir "$D" --n "$N" && \
python tools/kit_mc_madgraph.py predict "$P" --dir "$D" "$RUN" && \
python tools/kit_mc_madgraph.py metrics --dir "$D" --out "$D/kit_mc_madgraph_${P}.json" && echo "RUN_DONE"
