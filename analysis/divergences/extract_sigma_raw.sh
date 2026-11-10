#!/bin/bash
#SBATCH --job-name=ext_sig_raw
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=analysis/divergences/ext_sig_raw_%j.out
#SBATCH --error=analysis/divergences/ext_sig_raw_%j.out
set -e
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd $PROJECT_DIR
python analysis/divergences/extract_train_pool_sigma.py --process eeuu \
  --pool data_l1rawpool_eeuu/ee_uu_91-1000GeV_amplitudes.npy \
  --run_dir runs/eeuu_sigfit/raw_sigma \
  --out analysis/divergences/l1rawpool_sigma_eeuu.npz
echo "DONE extract_sigma_raw"
