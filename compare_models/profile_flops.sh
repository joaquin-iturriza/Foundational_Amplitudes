#!/bin/bash
#SBATCH --job-name=amp_flops
#SBATCH --partition=gpu_v100
#SBATCH --qos=gpu
#SBATCH --account=lpnhe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=compare_models/flops_%j.out
#SBATCH --error=compare_models/flops_%j.err
#
# Stage A of the lloca-vs-lgatr-vs-slim comparison: measure training FLOPs/step
# (fwd+bwd) for each architecture on the real 25-process recipe pipeline, matched
# to ~1.61M params. Output -> compare_models/flops.txt   SUBMIT: sbatch compare_models/profile_flops.sh

source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes

OUT=compare_models/flops.txt
: > "$OUT"
for M in lloca lgatr slim; do
  echo "############ profiling $M ############"
  rm -rf "compare_models/_probe_${M}"      # experiment aborts on an existing run_dir
  python compare_models/profile_flops.py "$M" 2>&1 | tee -a "$OUT"
done
echo "=== DONE; FLOPs/step ==="
grep FLOPS_STEP "$OUT"
