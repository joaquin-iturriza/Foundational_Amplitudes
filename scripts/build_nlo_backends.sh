#!/bin/bash
#SBATCH --job-name=build_nlo
#SBATCH --partition=htc
#SBATCH --account=lpnhe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
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
set -o pipefail   # the per-process verdict pipes through tee: report a crashed build as such

# Every entry of tools.nlo_virtual_pipeline.VIRT_PROCESSES, in table order.
PROCS=($(python -c "import sys; sys.path.insert(0, 'tools'); from nlo_virtual_pipeline import VIRT_PROCESSES; print(' '.join(VIRT_PROCESSES))"))   # the whole one-loop table

# Build + certify in parallel (independent standalone dirs; MadLoop builds are 1-core, ~1-2 GB).
PAR="${PAR:-$(( ${SLURM_CPUS_PER_TASK:-8} / 2 ))}"; [ "$PAR" -lt 1 ] && PAR=1
mkdir -p scripts/build_nlo_logs
one() {
  P="$1"
  if timeout 1800 python tools/nlo_virtual_pipeline.py "$P" --build --certify > "scripts/build_nlo_logs/$P.log" 2>&1; then
    v="$(grep -o "\[CERTIFY\] $P: \(PASS\|FAIL\)" "scripts/build_nlo_logs/$P.log" | tail -1 | sed 's/.*: //')"
    echo "$P ${v:-built, NO VERDICT}"
  else
    echo "$P BUILD/CERTIFY CRASHED or TIMED OUT"
  fi
}
export -f one
printf '%s\n' "${PROCS[@]}" | xargs -P "$PAR" -I{} bash -c 'one {}' > scripts/build_nlo_verdicts.txt
declare -A RESULT
while read -r P V; do RESULT[$P]="$V"; done < scripts/build_nlo_verdicts.txt
for P in "${PROCS[@]}"; do echo "######## $P"; grep -E "^\[(BUILD|CERTIFY)\]|  \[CERTIFY\]|DOUBLE|SINGLE|predicted|MadLoop  mean" "scripts/build_nlo_logs/$P.log" | tail -8; done

echo "===================== NLO BACKEND BUILD SUMMARY ====================="
for P in "${PROCS[@]}"; do
  printf "  %-10s %s\n" "$P" "${RESULT[$P]}"
done
echo "(verdicts are the certifier's own PASS/FAIL lines; anything else means no verdict was produced)"
