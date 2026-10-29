#!/bin/bash
# Build the laptop's CPU project environment .venv-cpu, for the tests (pytest + pytest-testmon).
# Same stack as the CC-IN2P3 .venv (python 3.11, torch 2.1.2, numpy 1.26.4, the rest pinned in
# scripts/cpu_env_requirements.txt from CC's pip freeze), CPU torch, no CUDA-only packages
# (xformers, triton): tests that need them run in a GPU job on a site. Idempotent: re-run to repair.
# Run tests that a change touches:  .venv-cpu/bin/pytest --testmon tests/ test_amp.py
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
V=.venv-cpu
[ -x "$V/bin/python" ] || "$UV" venv --python 3.11 --seed "$V"
export VIRTUAL_ENV="$PWD/$V"
"$UV" pip install -q --python "$V/bin/python" torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cpu
"$UV" pip install -q --python "$V/bin/python" -r scripts/cpu_env_requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
"$V/bin/python" -c "import torch, numpy, lloca.mup, torch_geometric, pytest, testmon; print('cpu env ok: python', __import__('sys').version.split()[0], 'torch', torch.__version__, 'numpy', numpy.__version__)"
