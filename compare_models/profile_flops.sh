#!/bin/bash
#SBATCH --job-name=amp_flops
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=compare_models/flops_%j.out
#SBATCH --error=compare_models/flops_%j.err
#SBATCH --gres=gpu:1
#
# Stage A of the lloca-vs-lgatr-vs-slim comparison: measure training FLOPs/step
# (fwd+bwd) for each architecture on the real 25-process recipe pipeline, matched
# to ~1.61M params. Output -> compare_models/flops.txt   SUBMIT: sbatch compare_models/profile_flops.sh

source "$(dirname "${BASH_SOURCE[0]:-$0}")/../sites/activate.sh"
cd "$PROJECT_DIR"

OUT=compare_models/flops.txt
: > "$OUT"
for M in lloca lgatr slim; do
  echo "############ profiling $M ############"
  rm -rf "compare_models/_probe_${M}"      # experiment aborts on an existing run_dir
  python compare_models/profile_flops.py "$M" 2>&1 | tee -a "$OUT"
done
echo "=== DONE; FLOPs/step ==="
grep FLOPS_STEP "$OUT"
