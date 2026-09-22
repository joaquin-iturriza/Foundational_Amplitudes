#!/bin/bash
#SBATCH --job-name=ev_deep
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=analysis/divergences/ev_deep_%j.out
#SBATCH --error=analysis/divergences/ev_deep_%j.out

set -e
module purge
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../sites/activate.sh"
cd "$PROJECT_DIR"

# Forward the COMMON deep-IR held-out test through both fine-tunes (uniform vs antenna),
# recovering each run's own preprocessing stats. Deep-IR-binned MSE(log|M|^2) is on the
# same absolute (de-standardized) scale, so the two samplings compare like-for-like.
python analysis/divergences/eval_heldout.py \
  --runs_root runs/pretrain22_heldout_uug \
  --run_prefix ft_deep_ --tags ${TAGS:-uniform,antenna,mixture} \
  --heldout analysis/divergences/uug_deep_test.npz \
  --out_prefix deep_eval_ \
  --summary deep_eval_summary.json

echo "DONE eval_deep"
