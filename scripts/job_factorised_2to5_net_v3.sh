#!/bin/bash
#SBATCH --job-name=f25net_v3
#SBATCH --cpus-per-task=5
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=f25net_v3_%j.out
#SBATCH --error=f25net_v3_%j.out
#
# The factorised head on catalog_v3's 40 2->5 probes with the network in place of the exact parent: each catalog_v3
# short arm's own prediction of the 2->4 parent (a training class) at the mapped events (tools/factorised_2to4_net.py
# --v3-nfinal 5, the maps of $SCRATCH/zz_factorised_v3_2to5). Arms: all four short runs (ARMS overrides, for a rerun of
# some). Output: $SCRATCH/zz_factorised_v3_2to5_net_<arm>/
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
for ARM in ${ARMS:-baseline dimensionless dimensionless_context dimensionless_twin}; do
    python tools/factorised_2to4_net.py --parent "$PROJECT_DIR/runs/catalog_v3_short/$ARM" --v3-nfinal 5 \
        --f24 "$SCRATCH/zz_factorised_v3_2to5" --out "$SCRATCH/zz_factorised_v3_2to5_net_$ARM" | sed -u "s/^/$ARM /"
done
echo F25NET_DONE
