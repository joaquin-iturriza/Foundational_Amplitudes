#!/bin/bash
#SBATCH --job-name=amp_prof_lg
#SBATCH --account=itg@v100
#SBATCH --partition=gpu_p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/compare_models/prof_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/compare_models/prof_%j.err

cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes
PY=/lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational/bin/python
$PY compare_models/profile_lgatr_ops.py
echo "PROFILE DONE"
