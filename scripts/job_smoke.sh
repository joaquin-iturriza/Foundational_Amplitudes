#!/bin/bash
#SBATCH --job-name=fa_smoke
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=runs/_logs/%x_%j.out
#SBATCH --error=runs/_logs/%x_%j.out
# Migration smoke test: proves the CC-IN2P3 chain end to end — scheduler accepts
# the directives, the venv runs on a compute node, CUDA is visible, and the
# project + lloca import there. Not a training run; safe to delete.
set -e
PROJ=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
cd "$PROJ"
PY=$WORK/conda/envs/foundational/bin/python
"$PY" - <<'PY'
import torch, numpy as np
print("torch", torch.__version__, "| numpy", np.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(2048, 2048, device="cuda")
    print("gpu matmul sum:", float((x @ x).sum().item()) != 0.0)
    print("host<->device roundtrip:", bool((x.cpu().numpy().shape == (2048, 2048))))
from lloca.mup import finalize, MuAdam
from models.lloca import LLOCAMuPTransformer
import experiment, run, preprocessing
print("project + lloca import on compute node: OK")
PY
