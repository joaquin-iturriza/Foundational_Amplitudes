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
# it was copied from, then the stored read's own scorer path. Prints the MSE, the mean predicted log|M|^2 and logF.
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
# the stored read's own path (FA-ccin2p3-f6f9251c @c2f90d83: tools/zero_shot_parent.py --parent <run>/sigma_stage
# --save-dir, default flags) on the same pool, at today's code; n_train 0: the train pools were deleted (D131), the
# val pool stands in for the probe's own target statistics (q does not depend on them: 2k against 10k, 0.360 both)
R="$O/dc_bbx_hh.yaml"
printf 'processes:\n- {name: dc_bbx_hh, sqrts: [13000, 13000], n_train: 0, n_val: 10000, n_test: 0}\n' > "$R"
python tools/zero_shot_parent.py --parent runs/catalog_norm_short/DlessInOrd/sigma_stage --save-dir "$O/zs" \
    --allow-physics-mismatch "job_evaluate_daniels: the stored read's pool" "$R" | grep -v ZERO_SHOT_HIST || exit 1
python -c "import numpy as np, sys; z = np.load(sys.argv[1]); q, t, F = z['q'], z['t'], z['logF']
print('EVAL_GPU zero_shot_parent mse', round(float(((q - t) ** 2).mean()), 4), 'mean_q', round(float((q - F).mean()), 4),
      'mean_t', round(float((t - F).mean()), 4), 'logF', np.unique(np.round(F, 5))[:3])" "$O/zs/dc_bbx_hh.npz"
echo EVAL_GPU_DONE
