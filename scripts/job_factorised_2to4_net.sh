#!/bin/bash
#SBATCH --job-name=factorised_2to4_net
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=factorised_2to4_net_%j.out
#SBATCH --error=factorised_2to4_net_%j.out
#
# D99, the level item's design (ii) with the network: DlessInOrdQc's own 2->3 prediction (each parent's c_p from the
# table) on the 40 2->4 probes' mapped events (tools/factorised_2to4_net.py; the mapped events and exact t3 from
# tools/factorised_2to4.py in $SCRATCH/zz_factorised_2to4). Output: $SCRATCH/zz_factorised_2to4_net/. Args: [arm] [extra]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
ARM="${1:-DlessInOrdQc}"; shift || true
RUN="$PROJECT_DIR/runs/catalog_norm_short/$ARM"
PAR="$RUN"; ls "$RUN"/sigma_stage/models/model_run0_best.pt* >/dev/null 2>&1 && PAR="$RUN/sigma_stage"
python tools/factorised_2to4_net.py --parent "$PAR" "$@"
