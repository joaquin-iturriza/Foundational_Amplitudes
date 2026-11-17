#!/bin/bash
#SBATCH --job-name=evaluate_daniels
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
# evaluate_daniels_catalog/evaluate.py on a GPU (xformers attention) on the first 2000 events of the dc_bbx_hh val pool,
# the events of the stored zero-shot read catalog_short_DlessInOrd_sigstage (GPU, 2026-10-23, MSE of log|M|^2 0.279,
# mean prediction 0.194 above the CPU evaluator's, whose MSE is 0.417): the committed model copy and the run directory
# it was copied from. Prints the MSE and the mean predicted log|M|^2 of each.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
P="$DATA_DIR/dc_bbx_hh_13000-13000GeV_val_lhe_amplitudes"
O="$SCRATCH/zz_evaluate_daniels_${SLURM_JOB_ID:-$$}"; mkdir -p "$O"
for M in evaluate_daniels_catalog/model runs/catalog_norm_short/DlessInOrd/sigma_stage; do
    python evaluate_daniels_catalog/evaluate.py --process bbx_hh --events "$P.npy" --alphas "$P.alphas.npy" --n 2000 \
        --model "$M" --out "$O/$(basename "$M").npz" || exit 1
    python -c "import numpy as np, sys; z = np.load(sys.argv[1]); q, t = z['logM2_pred'], z['logM2_true']
print('EVAL_GPU', sys.argv[2], 'mse', round(float(((q - t) ** 2).mean()), 4), 'mean_q', round(float(q.mean()), 4), 'mean_t', round(float(t.mean()), 4))" "$O/$(basename "$M").npz" "$M"
done
echo EVAL_GPU_DONE
