#!/bin/bash
#SBATCH --job-name=catalog_v3_short_prebuild
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=catalog_v3_short_prebuild_%j.out
#SBATCH --error=catalog_v3_short_prebuild_%j.out
#
# CPU: the memmap cache of one arm of scripts/job_catalog_v3_short.sh (its mode prebuild), built with the arm's own
# overrides and environment (LONG_HEADS, LONG_MEMMAP, GROW_K, ...), so the GPU run finds it. Sizing as
# scripts/job_mmap_prebuild.sh (16G covers the chunked build; format 2's whole-set build took about 20 min).
#   site submit ccin2p3 FA scripts/job_catalog_v3_short_prebuild.sh --item <item> --env LONG_HEADS=16 ... -- <arm>
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
exec bash "$_CCORCH_ROOT/scripts/job_catalog_v3_short.sh" "${1:?arm}" prebuild
