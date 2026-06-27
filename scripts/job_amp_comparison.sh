#!/bin/bash
#SBATCH --job-name=amp_f32_vs_f16
#SBATCH --partition=gpu_p2
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load anaconda-py3/2023.09
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational

cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

COMMON="data.dataset=[ee_ttbar_346-1000GeV_amplitudes] \
  data.amp_orders=[[0,0]] \
  training.iterations=50000 \
  training.batchsize=512 \
  training.log_every_n_steps=200 \
  training.validate_every_n_steps=2000 \
  seed=42 \
  evaluate=true \
  plot=true"

echo "=== float32 ==="
python run.py $COMMON training.float16=false run_name=amp_compare_f32

echo "=== float16 ==="
python run.py $COMMON training.float16=true  run_name=amp_compare_f16
