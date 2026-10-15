#!/bin/bash
#SBATCH --job-name=zero_shot_parent
#SBATCH --cpus-per-task=3
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=zero_shot_parent_%j.out
#SBATCH --error=zero_shot_parent_%j.out
#
# Real zero-shot of one or more pretrainings on the multiplicity probes (D55; tools/zero_shot_parent.py), each built
# from the pretraining's own config: u dbar -> W+ 4g (one emission past rung 9's W+3g) and the Z+ng probes.
#   site submit <site> FA scripts/job_zero_shot_parent.sh --env PARENTS=runs/tp3_jcaus_r9/trial_0073,runs/tp3_gstd_r9/trial_0073
# PARENTS: comma-separated pretraining run dirs (relative to the checkout or absolute; also accepted as arguments).
# PROBES: comma-separated probe names (default below), each recipes/transfer_probe_<probe>.yaml.

_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
PROBES=${PROBES:-udbar_Wgggg,uubar_Zg,uubar_Zgg,uubar_Zggg,uubar_Zgggg,uubar_Zggggg}
recipes=$(for p in ${PROBES//,/ }; do echo recipes/transfer_probe_$p.yaml; done)
for parent in ${PARENTS//,/ } "$@"; do
  [ -f "$parent/data_stats.json" ] || { echo "ZERO_SHOT_MISSING $parent"; continue; }
  python tools/zero_shot_parent.py --parent "$parent" $recipes
done
