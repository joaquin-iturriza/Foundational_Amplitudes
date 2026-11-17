#!/bin/bash
#SBATCH --job-name=layer_dump
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
# evaluate_daniels_catalog/layer_dump.py on the first 2000 events of the dc_bbx_hh val pool: every module's output of
# the evaluator's forward pass on a GPU with xformers and TF32 as configured, TF32 off, xformers hidden (the dense masked
# attention the CPU runs) with TF32 off, and on the CPU, to find where the CPU read (0.194 low in log|M|^2) parts.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
set -o pipefail
P="$DATA_DIR/dc_bbx_hh_13000-13000GeV_val_lhe_amplitudes.npy"
O="$SCRATCH/zz_layer_dump_${SLURM_JOB_ID:-$$}"; mkdir -p "$O"
D="python evaluate_daniels_catalog/layer_dump.py --pool $P"
$D --out "$O/gpu_tf32.npz" 2>&1 | tee "$O/gpu_tf32.log" | grep -E 'LAYER_DUMP|Error' || exit 1
$D --out "$O/gpu_notf32.npz" --no-tf32 2>&1 | tee "$O/gpu_notf32.log" | grep -E 'LAYER_DUMP|Error' || exit 1
$D --out "$O/gpu_dense_notf32.npz" --no-tf32 --no-xformers 2>&1 | tee "$O/gpu_dense_notf32.log" | grep -E 'LAYER_DUMP|Error' || exit 1
CUDA_VISIBLE_DEVICES= $D --out "$O/cpu.npz" --no-xformers 2>&1 | tee "$O/cpu.log" | grep -E 'LAYER_DUMP|Error' || exit 1
for AB in "gpu_notf32 cpu" "gpu_dense_notf32 cpu" "gpu_tf32 gpu_notf32" "gpu_tf32 cpu"; do
    set -- $AB; echo "COMPARE $1 vs $2"
    python evaluate_daniels_catalog/layer_dump.py --compare "$O/$1.npz" "$O/$2.npz" | awk 'NR % 10 == 0 || $3 > 1e-3 || /FIRST_DIFF/' | head -60
done
echo "LAYER_DUMP_DIR $O"
echo LAYER_DUMP_DONE
