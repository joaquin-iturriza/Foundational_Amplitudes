#!/bin/bash
#SBATCH --job-name=amp_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source "$(dirname "${BASH_SOURCE[0]:-$0}")/../sites/activate.sh"

cd "$PROJECT_DIR"
python run.py training.iterations=1000 plot=false
