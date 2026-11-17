#!/bin/bash
#SBATCH --job-name=lap_pe_table
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
# The diagram encoder's Laplacian PEs of every catalog diagram computed on a GPU node, where the evaluator's model reads
# as stored (FA-lxplus-fee0d906, FA-ccin2p3-c3e63032: MSE 0.2791 on 2000 dc_bbx_hh val events), then the evaluator on
# this node's CPU with that table (expected 0.2791) and the table printed (gzip, base64) to be committed as
# evaluate_daniels_catalog/model/lap_pe.json.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
set -o pipefail
O="$SCRATCH/zz_lap_pe_${SLURM_JOB_ID:-$$}"; mkdir -p "$O"
P="$DATA_DIR/dc_bbx_hh_13000-13000GeV_val_lhe_amplitudes"
python evaluate_daniels_catalog/lap_pe_table.py --out "$O/lap_pe.json" || exit 1
EVALUATE_LAP_PE="$O/lap_pe.json" CUDA_VISIBLE_DEVICES= python evaluate_daniels_catalog/evaluate.py --process bbx_hh \
    --events "$P.npy" --alphas "$P.alphas.npy" --n 2000 --out "$O/cpu.npz" 2>&1 | tee "$O/cpu.log" | grep -E '^evaluate|Error' || exit 1
echo "__B64__"; gzip -c "$O/lap_pe.json" | base64 -w0; echo
echo LAP_PE_DONE
