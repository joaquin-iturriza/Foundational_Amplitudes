#!/bin/bash
#SBATCH --job-name=zero_shot_timing
#SBATCH --cpus-per-task=3
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=zero_shot_timing_%j.out
#SBATCH --error=zero_shot_timing_%j.out
#
# A timing read of the zero-shot reader (tools/zero_shot_parent.py) on N probes of one held-out axis of a catalog short
# arm, as the arm's own axis loop reads them (scripts/job_catalog_v3_short.sh: one recipe per process, --signed-magnitude,
# the axis's split), into a folder of this job's own; then each probe's prediction is compared with the arm's earlier
# read of it ($SCRATCH/zz_<cat>_short_<arm>/<axis>/<probe>.npz), array by array: ZS_SAME / ZS_DIFF / ZS_NO_EARLIER per
# probe and one ZS_COMPARE summary line. Every ZERO_SHOT record carries s_stages (seconds per stage of that probe).
#   REREAD=<run id> site submit <site> FA scripts/job_zero_shot_timing.sh --env ARM=v4_dimensionless_muon --env AXIS=H1 --env N=50
# ARM, CAT (v4), AXIS (H1), N (50), SPLIT (val; the axis loop's split for the axis).

_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
ARM=${ARM:?ARM: the short arm whose run is read}; CAT=${CAT:-v4}; AXIS=${AXIS:-H1}; N=${N:-50}; SPLIT=${SPLIT:-val}
PARENT="$PROJECT_DIR/runs/catalog_${CAT}_short/$ARM"; EARLIER="$SCRATCH/zz_${CAT}_short_$ARM/$AXIS"
ls "$PARENT"/models/model_run0_best.pt* >/dev/null 2>&1 || { echo "no best checkpoint in $PARENT"; exit 1; }
R="$SCRATCH/zz_zs_timing_${ARM}_${AXIS}_${SLURM_JOB_ID:-$$}"
mkdir "$R" "$R/recipes" "$R/pred" || { echo "$R exists: refusing a folder this job did not make"; exit 1; }
echo "ZS_TIMING parent=$PARENT axis=$AXIS n=$N split=$SPLIT code=$(git rev-parse --short HEAD) host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
# the first N processes of the axis with a pool of the split, one recipe each (the axis file's top-level keys kept)
python - "recipes/catalog_${CAT}_heldout_${AXIS}.yaml" "$R/recipes" "$N" "$SPLIT" <<'PY' || exit 1
import glob, os, sys, yaml
r = yaml.safe_load(open(sys.argv[1])); top = {k: v for k, v in r.items() if k != "processes"}
have = lambda n: glob.glob(os.path.join(os.environ["DATA_DIR"], f"{n}_*_{sys.argv[4]}_*amplitudes.npy"))
ps = [p for p in r["processes"] if have(p["name"])][:int(sys.argv[3])]
for p in ps:
    yaml.safe_dump({**top, "processes": [p]}, open(f"{sys.argv[2]}/{p['name']}.yaml", "w"))
print(f"ZS_TIMING_PROBES {len(ps)} of {len(r['processes'])}")
PY
T0=$(date +%s)
python tools/zero_shot_parent.py --parent "$PARENT" --save-dir "$R/pred" --split "$SPLIT" --signed-magnitude "$R"/recipes/*.yaml || exit 1
echo "ZS_TIMING_WALL $(( $(date +%s) - T0 )) s"
python - "$R/pred" "$EARLIER" <<'PY'
import glob, os, sys
import numpy as np
new, old = sys.argv[1], sys.argv[2]
same = diff = none = 0
for f in sorted(glob.glob(os.path.join(new, "*.npz"))):
    name = os.path.basename(f)[:-4]; g = os.path.join(old, name + ".npz")
    if not os.path.exists(g):
        none += 1; print(f"ZS_NO_EARLIER {name}"); continue
    a, b = np.load(f), np.load(g)
    bad = {}
    for k in ("t", "q", "q_own", "q_aff", "h", "sig"):
        if k in a.files and k in b.files and not np.array_equal(a[k], b[k]):
            x, y = a[k], b[k]
            bad[k] = (float(np.max(np.abs(x - y))) if x.shape == y.shape and x.size else f"shape {x.shape} vs {y.shape}")
    if bad:
        diff += 1; print(f"ZS_DIFF {name} {bad}")
    else:
        same += 1; print(f"ZS_SAME {name}")
print(f"ZS_COMPARE same={same} diff={diff} no_earlier={none} earlier={old}")
PY
