#!/bin/bash
#SBATCH --job-name=build_nlo
#SBATCH --partition=prepost
#SBATCH --account=itg@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=10:00:00
#SBATCH --output=scripts/build_nlo_%j.out
#SBATCH --error=scripts/build_nlo_%j.err
#
# Build + pole-certify the MadLoop standalones for the NLO-virtual process set on
# the FREE prepost partition (CPU, weight 0 — no GPU budget). 2->2 qqbar are fast
# and robust; the 2->3 qqg loops are heavier and may be slow/fail — each is
# attempted independently and the PASS/FAIL is reported so the recipe can use
# whatever certifies.

module load anaconda-py3/2023.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/fswork/projects/rech/itg/ulm49ia/conda/envs/foundational
cd /lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes

# New backends to build (ee_ss / ee_ttbar already built). 2->2 first (cheap), then 2->3.
PROCS=(ee_uu ee_dd ee_cc ee_bb ee_uug ee_ddg)

declare -A RESULT
for P in "${PROCS[@]}"; do
  echo "######################## BUILD+CERTIFY $P ########################"
  if timeout 7200 python tools/nlo_virtual_pipeline.py "$P" --build --certify; then
    RESULT[$P]="built (see CERTIFY line above for PASS/FAIL)"
  else
    RESULT[$P]="BUILD/CERTIFY FAILED or TIMED OUT"
  fi
done

echo "===================== NLO BACKEND BUILD SUMMARY ====================="
for P in "${PROCS[@]}"; do
  printf "  %-10s %s\n" "$P" "${RESULT[$P]}"
done
echo "Look for '[CERTIFY] <proc>: PASS/FAIL' lines above for the pole check verdict."
