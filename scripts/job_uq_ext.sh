#!/bin/bash
#SBATCH --job-name=uq_ext
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --gpus=1
# zng_uq_benchmark: the networks of lloca-experiments-uq (branch uq, pinned commit) trained and evaluated by their own
# job scripts (uq_ops/jobs/<job>, unchanged) on the converted Zenodo 16793011 files of job_zng_fetch.sh.
#     site submit ccin2p3 FA scripts/job_uq_ext.sh --item zng_uq_benchmark -- <job> <name> [VAR=value ...]
# <job>   time_check.sh (their pre-production timing: each network alone, then all four on one GPU, 2000 steps each)
#         or train_shared.sh (their production: one multiplicity, all four networks on the GPU; RUN_N=1..4)
# <name>  the run folder $WORK/ext/uq_runs/runs/<name> (their UQ_OUT); VAR=value are exported (RUN_N, RUN_EXP, ...)
# First use builds their environment (their uq_ops/requirements-pinned.txt, torch from the cu126 index so V100 and
# H100 both run it; flash_attn and xformers left out: the amplitude code imports neither) and runs their tests.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
set -euo pipefail
JOB="$1"; NAME="$2"; shift 2
COMMIT=1074f6abd4998b2484e142217f3a245a40ee057c
EXT="$WORK/ext"; REPO="$EXT/lloca-experiments-uq"; VENV="$EXT/uq_venv"; BASE="$EXT/uq_runs"; DATA="$EXT/uq_data"
mkdir -p "$EXT" "$BASE/runs"
if [ ! -d "$REPO/.git" ]; then git clone -q -b uq https://github.com/supriodubey/lloca-experiments-uq.git "$REPO"; fi
git -C "$REPO" fetch -q origin uq && git -C "$REPO" checkout -q "$COMMIT"
if [ ! -x "$VENV/bin/python" ] || [ ! -e "$VENV/.ok_$COMMIT" ]; then
    PY311="$(python -c 'import sys; print(sys.base_prefix)')/bin/python3.11"
    "$PY311" -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
    grep -v -E '^(torch|torchvision|triton|nvidia-[a-z0-9-]+|flash_attn|xformers)==' "$REPO/uq_ops/requirements-pinned.txt" \
        > "$EXT/req_noflash.txt"
    "$VENV/bin/pip" install -q -r "$EXT/req_noflash.txt"
    "$VENV/bin/pip" install -q --no-deps -e "$REPO"
    (cd "$REPO" && "$VENV/bin/python" -m pytest -q tests/uq tests/experiments/test_amp_uq.py) && touch "$VENV/.ok_$COMMIT"
fi
# their layout <UQ_DATA>/<proc>/<proc>_<i>.npy, linked to the files job_zng_fetch.sh extracted
for p in zg zgg zggg zgggg; do mkdir -p "$DATA/$p"; for i in 0 1 2; do
    f=$(find "$DATA_DIR/zng_uq/raw" -name "${p}_${i}.npy" | head -1); [ -n "$f" ] || { echo "no ${p}_${i}.npy"; exit 3; }
    ln -sf "$f" "$DATA/$p/${p}_${i}.npy"; done; done
for kv in "$@"; do export "$kv"; done
export UQ_SITE=ccin2p3 UQ_REPO="$REPO" UQ_PY="$VENV/bin/python" UQ_BASE="$BASE" UQ_DATA="$DATA" \
    UQ_OUT="$BASE/runs/$NAME" UQ_JOBS="$REPO/uq_ops/jobs"
mkdir -p "$UQ_OUT"
echo "uq_ext job=$JOB out=$UQ_OUT commit=$(git -C "$REPO" log -1 --format=%h) gpu=$(nvidia-smi -L | head -1)"
bash "$UQ_JOBS/$JOB"
echo "RUN_DONE uq_ext $JOB $NAME"
