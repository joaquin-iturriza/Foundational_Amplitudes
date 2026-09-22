#!/bin/bash
#SBATCH --job-name=scan_ab_bo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=compare_models/scan_ab_bo_%A_%a.out
#SBATCH --error=compare_models/scan_ab_bo_%A_%a.err
#
# Proper DyHPO HPO for the 3-way coupling+mass A/B (mirrors encab_bo_run.sh):
# one GPU job per arm, 15 trials looped SEQUENTIALLY so every observe() lands
# before the next suggest() (informed surrogate). Each arm's sweep optimises the
# FULL HP space (lr, regularization_lambda, warmup, eta_min, ema_decay) at 5000
# steps. Compare best-vs-best val_loss_no_reg. Data is prebuilt (require_cache),
# so no prepost prebuild dependency. Pre-init: make_scan_ab_sweeps.py + generate_sweep.

set -e
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../sites/activate.sh"
cd "$PROJECT_DIR"
PROJ=$PWD
export OMP_NUM_THREADS=1

ARMS=(off scalar diagram)
NAME=${ARMS[$SLURM_ARRAY_TASK_ID]}
CFG="$PROJ/sweeps/scan_ab/scan_ab_${NAME}/sweep_config.yaml"
N_TRIALS=15

echo "=== arm $NAME : sequential DyHPO, $N_TRIALS trials, $(date +%H:%M:%S) ==="
for i in $(seq 0 $((N_TRIALS-1))); do
  echo "----- [$NAME] trial $i  $(date +%H:%M:%S) -----"
  python "$PROJ/sweep/run_trial.py" --sweep-config "$CFG" --trial-idx "$i" \
    || echo ">>> [$NAME] trial $i FAILED (continuing)"
done
echo "=== arm $NAME DONE $(date +%H:%M:%S) ==="
