#!/bin/bash
#SBATCH --job-name=mixed_order_virt
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=mixed_order_virt_%j.out
#SBATCH --error=mixed_order_virt_%j.out
#
# CPU only (catalog_coverage, the user's "c) Investigate it"): why eight catalog_v4 one-loop entries fail the pole
# check. tools/mixed_order_virt_check.py builds each born as [virt=QCD] with and without QED<=n, one squared order at a
# time, [sqrvirt=QCD], [virt=QCD QED] and an aMC@NLO check_poles, reads MadLoop's answer per squared order, and scans
# MadLoopParams (stability threshold, quadruple precision, reduction library). Everything under $SCRATCH/zz_mixed_order_virt.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
python tools/mixed_order_virt_check.py run --workers "${SLURM_CPUS_PER_TASK:-8}" ${MOV_ARGS:-}
