#!/bin/bash
#SBATCH --job-name=v4_hwpwm_wt0
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=v4_hwpwm_wt0_%j.out
#SBATCH --error=v4_hwpwm_wt0_%j.out
#
# CPU only (plan catalog_coverage, the user 2026-10-26, D154): the one-loop b b~ > h w+ w- (v4_bbx_hwpwm_nlo) at zero
# top width (recipes/gen_catalog_v4.py WIDTH_OVERRIDE), every other entry at 1.4915.
#   1. its standalone's card re-patched to the override (build_virt_standalone; no recompile, MadLoop reads the card)
#   2. the cards read back (the width MadGraph uses: a column-1 DECAY 6 line, else the compiled default): 0 for
#      v4_bbx_hwpwm, 1.4915 for its neighbour v4_bbx_tapvtwm
#   3. tools/pole_guard_diag.py over its window [286, 6000] GeV at the card width (now zero): refused fraction against
#      the generation guard's 2% limit
#   4. only if 2 and 3 pass: its pool build (scripts/job_catalog_v4_regen.sh, V4REGEN_ONLY, one shard)
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
python - <<'EOF' || { echo "WT0_CARD_FAILED"; exit 1; }
import os, re, sys
sys.path.insert(0, "tools")
import nlo_virtual_pipeline as V
V.build_virt_standalone("v4_bbx_hwpwm")
bad = 0
def col1_wt(card):
    """the top width MadGraph reads from a card: its column-1 DECAY 6 line (an indented one is skipped)"""
    lines = [l.rstrip("\n") for l in open(card) if re.match(r"\s*DECAY\s+6\s", l, re.I)]
    col1 = [l for l in lines if l.upper().startswith("DECAY")]
    return (float(col1[0].split()[2]) if len(col1) == 1 else None), lines
for name, want in (("v4_bbx_hwpwm", 0.0), ("v4_bbx_tapvtwm", 1.4915)):
    cards = os.path.join(V.virt_standalone_dir(name), "Cards")
    got, lines = col1_wt(os.path.join(cards, "param_card.dat"))
    src = "card"
    if got is None:           # no column-1 line: the compiled default, which is param_card_default.dat's
        got, _ = col1_wt(os.path.join(cards, "param_card_default.dat")); src = "compiled default"
    ok = got is not None and abs(got - want) < 1e-9
    bad += not ok
    print(f"WT0_CARD {'OK ' if ok else 'BAD'} {name} want {want} effective {got} ({src}) lines {lines!r}")
sys.exit(1 if bad else 0)
EOF
DIAGF="$SCRATCH/zz_pole_guard_wt0_${SLURM_JOB_ID:-$$}.txt"
python tools/pole_guard_diag.py v4_bbx_hwpwm --sqrts-min 286 --sqrts-max 6000 --n 1000 | tee "$DIAGF"
R=$(awk '/^DIAG /{for(i=1;i<=NF;i++) if($i ~ /^redrawn=/){split($i,a,"="); print a[2]}}' "$DIAGF")
echo "WT0_DIAG redrawn=${R:-none} limit=0.02"
[ -n "$R" ] && awk -v r="$R" 'BEGIN{exit !(r <= 0.02)}' || { echo "WT0_DIAG_FAILED"; exit 1; }
rm -f "$DIAGF"
V4REGEN_ONLY=v4_bbx_hwpwm_nlo V4REGEN_N=1 V4REGEN_K=0 SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}" \
    bash scripts/job_catalog_v4_regen.sh
