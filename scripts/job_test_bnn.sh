#!/bin/bash
#SBATCH --job-name=test_bnn
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --output=test_bnn_%j.out
#SBATCH --error=test_bnn_%j.out
#
# Unit tests of the BNN last layer (model.net.bayesian_last_layer, tests/test_bayesian_readout.py): the variational
# μP readout (init, KL, local reparametrization, μP infshapes/rescaling/MuAdam), and on the full stack the off path
# bit for bit, the readout swap, and experiment._reg_term (KL / N_train) on the CUDA/xformers path.

_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python -c "import torch, xformers; print('torch', torch.__version__, 'xformers', xformers.__version__, 'cuda', torch.cuda.is_available())"
python tests/test_bayesian_readout.py
