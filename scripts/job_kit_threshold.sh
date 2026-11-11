#!/bin/bash
#SBATCH --job-name=kit_thr
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# kit_benchmark threshold scan (tools/kit_threshold_scan.py nlo): MG5_aMC fixed-order LO and NLO e+ e- > t t~ [QCD]
# cross sections at the scan's sqrt s values i = k, k+K, k+2K, ... (part k of K), each part in its own MG5_aMC output
# dir runs/kit/threshold_scan/amc_<k> (one output per part: launches in one dir cannot run at once). CPU only.
#     site submit ccin2p3 FA scripts/job_kit_threshold.sh --item kit_benchmark -- <k> <K> [--only <i>]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
K_="$1"; N_="$2"; shift 2
python tools/kit_threshold_scan.py nlo --part "$K_/$N_" "$@" && echo "RUN_DONE"
