#!/bin/bash
# Build the small local Python environment sweep/drive.py needs (the DyHPO sampler imports
# torch, scikit-learn, scipy, numpy, yaml). CPU torch only; nothing trains on the laptop.
# Idempotent: re-run to repair. Used automatically by sweep/drive.py when it finds itself
# running without torch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
V=.venv-driver
if [ ! -x "$V/bin/python" ]; then
  python3 -m venv "$V" 2>/dev/null || python3 -m venv --without-pip "$V"
fi
PY="$V/bin/python"
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  # WSL has no python3-venv/ensurepip: bootstrap pip (Windows curl is 100x faster than WSL's)
  tmp=$(mktemp)
  if [ -x /mnt/c/Windows/System32/curl.exe ]; then /mnt/c/Windows/System32/curl.exe -sSL -o "$tmp" https://bootstrap.pypa.io/get-pip.py
  else curl -sSL -o "$tmp" https://bootstrap.pypa.io/get-pip.py; fi
  "$PY" "$tmp" -q; rm -f "$tmp"
fi
"$PY" -m pip install -q scikit-learn scipy "numpy<2" pyyaml gpytorch   # numpy<2: the DyHPO code uses np.NINF
"$PY" -c "import torch" 2>/dev/null || "$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
"$PY" -c "import torch, sklearn, scipy, yaml, gpytorch; print('driver env ok: torch', torch.__version__, 'sklearn', sklearn.__version__)"
