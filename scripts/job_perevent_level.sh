#!/bin/bash
#SBATCH --job-name=perevent_level
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
# CPU only (llm_compare, D566): analysis/catalog_daniel/perevent_level.py over catalog_daniel's cached train pools and
# diagram sidecars on CC. Its own job, not `site run --job`, for the memory (~2.7M events x 43 features, the CPU block's
# 2G is too small).
#     site submit <site with the pools: lxplus> FA scripts/job_perevent_level.sh --item llm_compare [--env PEL_MODEL=mlp]
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
M=${PEL_MODEL:-hgb}; J=analysis/catalog_daniel/data/perevent_level$([ "$M" = mlp ] && echo _mlp).json
OMP_NUM_THREADS=4 python analysis/catalog_daniel/perevent_level.py --model "$M" && cat "$J" \
  | python -c "import json,sys; print('PEL_JSON', json.dumps(json.load(sys.stdin)))" && echo "RUN_DONE"
