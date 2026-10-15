#!/bin/bash
#SBATCH --job-name=test_causal
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --output=test_causal_%j.out
#SBATCH --error=test_causal_%j.out
#
# CPU + GPU unit tests of the causal (autoregressive) attention mask (D55, tests/test_causal_mask.py): mask shape,
# frames-net edges, the prefix-leakage test through the dense path and through xformers on CUDA, the particle order,
# the affine multiplicity standardization.

_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python -c "import torch, xformers; print('torch', torch.__version__, 'xformers', xformers.__version__, 'cuda', torch.cuda.is_available())"
python tests/test_causal_mask.py
