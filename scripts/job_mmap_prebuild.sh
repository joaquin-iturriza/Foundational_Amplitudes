#!/bin/bash
#SBATCH --job-name=mmap_prebuild
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=mmap_prebuild_%j.out
#SBATCH --error=mmap_prebuild_%j.out
#
# Builds the shared memmap cache (data_mmap.py, format 3) of a run's pools on CPU, so the GPU jobs only open it
# (a GPU run without its cache refuses). Arguments: the run's own Hydra overrides, as its job passes them to run.py;
# a cache built for larger caps serves every run with smaller ones, and no seed or statistic enters it, so one
# prebuild serves every seed and arm on the same recipe. Idempotent: a cache already built is left as it is.
#   sbatch scripts/job_mmap_prebuild.sh data.processes_file=recipes/catalog_v4_train.yaml data.train_subsample=100000 ...
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
PYTHONPATH=. python run.py "$@" ++data.memmap=true ++data.memmap_prebuild=true \
    exp_name=zz_mmap_prebuild run_name="prebuild_${SLURM_JOB_ID:-$$}" training.num_workers=0
