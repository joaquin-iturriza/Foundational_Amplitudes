#!/bin/bash
#SBATCH --job-name=linear_readout
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=linear_readout_%j.out
#SBATCH --error=linear_readout_%j.out
#
# D108: linear read-out of rungs 6 and 7's frozen trunk features on u ubar -> Z g, every D of the 8k grid
# (tools/linear_readout.py). Inference and a ridge fit only; one GPU, minutes.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
# with arguments: those sweep configs (e.g. every rung's ee_Za cells, the no-gain read-out of 2026-10-25)
CFGS=("$@")
if [ ${#CFGS[@]} -eq 0 ]; then
  for r in 6 7; do for k in 2 3 4 5 6 7 8; do CFGS+=("sweep/sweep_config_tp3_r${r}fte_uubar_Zg_d${k}.yaml"); done; done
fi
python tools/linear_readout.py --out "$SCRATCH/zz_linear_readout" "${CFGS[@]}"
