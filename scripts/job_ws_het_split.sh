#!/bin/bash
#SBATCH --job-name=ws_het_split
#SBATCH --cpus-per-task=5
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=ws_het_split_%j.out
#SBATCH --error=ws_het_split_%j.out
#
# Plan entry warm_start step (a) (D291): the het NLL of each trial of the k=533 warm-start sweep ws_c1_k533_hpo8 (and of
# the untuned k=533 warm start and the parent) split into its 1/sigma^2-weighted squared-error term and its log sigma
# term on the trials' own validation split; analysis/catalog_v4/ws_het_split.py run (one WS_SPLIT line per model).
#   site submit ccin2p3 FA scripts/job_ws_het_split.sh --hdr=--mem=16G --item warm_start
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
echo "WS_HET_SPLIT code=$(git rev-parse --short HEAD) host=$(hostname)"
T0=$(date +%s)
python analysis/catalog_v4/ws_het_split.py run; rc=$?
echo "RUN_SUMMARY {\"rc\": $rc, \"wall_s\": $(( $(date +%s) - T0 ))}"
exit $rc
