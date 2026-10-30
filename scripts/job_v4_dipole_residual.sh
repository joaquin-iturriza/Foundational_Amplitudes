#!/bin/bash
#SBATCH --job-name=v4_dipole_residual
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=v4_dipole_residual_%j.out
#SBATCH --error=v4_dipole_residual_%j.out
#
# CPU only (plan v3_prep_explore, the dipole design): the per-event soft-dipole factor against the control pair's
# zero-shot reads already on the site ($SCRATCH/zz_v4_short_<arm>/{H2,TR}), momenta from the pools' frozen rows
# (analysis/catalog_v4/dipole_residual.py). One output per read directory under $SCRATCH/zz_v4_dipole_<job>; the
# summary lines are printed as DIPOLE_RESIDUAL {json}.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
OUT="$SCRATCH/zz_v4_dipole_${SLURM_JOB_ID:-$$}"; mkdir -p "$OUT"
for arm in v4_dimensionless_muon v4_dimensionless_muon_seed2; do
    for ax in H2 TR; do
        d="$SCRATCH/zz_v4_short_$arm/$ax"
        [ -d "$d" ] || { echo "missing $d"; exit 1; }
        s=$(python analysis/catalog_v4/dipole_residual.py "$d" --out "$OUT/${arm}_$ax.json") || exit 1
        echo "DIPOLE_RESIDUAL {\"arm\": \"$arm\", \"axis\": \"$ax\", \"summary\": $s}"
    done
done
echo "V4_DIPOLE_DONE $OUT"
