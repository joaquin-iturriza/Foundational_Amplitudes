#!/bin/bash
#SBATCH --job-name=build_nlo
#SBATCH --partition=htc
#SBATCH --account=lpnhe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10:00:00
#SBATCH --output=scripts/build_nlo_%j.out
#SBATCH --error=scripts/build_nlo_%j.err
#
# Build + pole-certify the MadLoop standalones for the NLO-virtual process set on
# the FREE prepost partition (CPU, weight 0 — no GPU budget). 2->2 qqbar are fast
# and robust; the 2->3 qqg loops are heavier and may be slow/fail — each is
# attempted independently and the PASS/FAIL is reported so the recipe can use
# whatever certifies.

source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/.venv/bin/activate
source /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes/scripts/env_ccin2p3.sh
cd /sps/lpnhe/jiturrizaramirez01/Foundational_Amplitudes

# Every entry of tools.nlo_virtual_pipeline.VIRT_PROCESSES, in table order.
PROCS=($(python -c "import sys; sys.path.insert(0, 'tools'); from nlo_virtual_pipeline import VIRT_PROCESSES; print(' '.join(VIRT_PROCESSES))"))   # the whole one-loop table

declare -A RESULT
for P in "${PROCS[@]}"; do
  echo "######################## BUILD+CERTIFY $P ########################"
  if timeout 1800 python tools/nlo_virtual_pipeline.py "$P" --build --certify | tee "scripts/build_nlo_${P}.log"; then
    RESULT[$P]="$(grep -o "\[CERTIFY\] $P: \(PASS\|FAIL\)" "scripts/build_nlo_${P}.log" | tail -1 | sed 's/.*: //')"
    [ -z "${RESULT[$P]}" ] && RESULT[$P]="built, NO VERDICT"
  else
    RESULT[$P]="BUILD/CERTIFY CRASHED or TIMED OUT"
  fi
  rm -f "scripts/build_nlo_${P}.log"
done

echo "===================== NLO BACKEND BUILD SUMMARY ====================="
for P in "${PROCS[@]}"; do
  printf "  %-10s %s\n" "$P" "${RESULT[$P]}"
done
echo "(verdicts are the certifier's own PASS/FAIL lines; anything else means no verdict was produced)"
