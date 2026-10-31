#!/bin/bash
#SBATCH --job-name=zero_shot_timing
#SBATCH --cpus-per-task=5
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
# ARM, CAT (v4), AXIS (H1), N (50), SPLIT (val; the axis loop's split for the axis), PROFILE (K: the first K probes
# again under cProfile, ZS_PROF lines), AB (1: the A/B of the reader's paths below; AB_PATHS picks them). Ends with one RUN_SUMMARY line (rc, probes, wall seconds).

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
if [ -n "$AB" ]; then
    # AB=1: the reader's paths on the same probes, GPU and job: A the original (config re-read per probe, one process),
    # A2 A again (the GPU forward's own run-to-run difference), B the config parsed once (one process), C B with
    # ZS_WORKERS (default 4) processes; each prediction of A2, B and C compared array by array with A's (ZS_AB_SAME /
    # ZS_AB_DIFF, one ZS_AB summary per path, with the largest difference per array)
    # AB_PATHS: the paths, each "name build workers parent_copy" (parent_copy deep: the parent checkpoint deep-copied
    # per probe as before 2026-10-31; shared: its containers copied, its tensors shared); the first is the reference
    AB_PATHS=${AB_PATHS:-"A load 1 deep;A2 load 1 deep;B cached 1 deep;C cached ${ZS_WORKERS:-4} deep"}
    IFS=';' read -ra PATHS <<< "$AB_PATHS"; REF=$(echo "${PATHS[0]}" | awk '{print $1}')
    for P in "${PATHS[@]}"; do
        set -- $P; mkdir "$R/pred_$1" || exit 1; T1=$(date +%s)
        ZS_PARENT_COPY=$4 python tools/zero_shot_parent.py --parent "$PARENT" --save-dir "$R/pred_$1" --split "$SPLIT" --signed-magnitude \
            --build "$2" --workers "$3" "$R"/recipes/*.yaml > "$R/read_$1.log" 2>&1
        rc=$?; echo "ZS_AB_WALL path=$1 build=$2 workers=$3 parent_copy=$4 rc=$rc wall_s=$(( $(date +%s) - T1 ))"
        grep -E "^(ZS_WORKERS|ZERO_SHOT \{)" "$R/read_$1.log" | sed "s/^/$1 /"
        [ "$rc" = 0 ] || { tail -40 "$R/read_$1.log"; echo "RUN_SUMMARY zero_shot_timing rc=$rc stage=read_$1"; exit "$rc"; }
    done
    python - "$R" "$REF" "${PATHS[@]}" <<'PY'
import glob, os, sys
import numpy as np
R, ref = sys.argv[1], sys.argv[2]
for path in [p.split()[0] for p in sys.argv[3:]][1:]:
    same = diff = 0; mx = {}
    for f in sorted(glob.glob(os.path.join(R, f"pred_{ref}", "*.npz"))):
        name = os.path.basename(f); g = os.path.join(R, f"pred_{path}", name)
        a = np.load(f); b = np.load(g) if os.path.exists(g) else None
        bad = {k: "missing" for k in (["npz"] if b is None else [k for k in a.files if k not in b.files])}
        for k in ([] if b is None else a.files):
            if k in b.files and not np.array_equal(a[k], b[k]):
                bad[k] = float(np.max(np.abs(a[k].astype(float) - b[k].astype(float)))) if a[k].shape == b[k].shape and a[k].dtype.kind in "fiub" else "differs"
        for k, v in bad.items():
            if isinstance(v, float):
                mx.setdefault(k, []).append(v)
        if bad:
            diff += 1; print(f"ZS_AB_DIFF {path} {name[:-4]} {bad}")
        else:
            same += 1
    print(f"ZS_AB path={path} vs={ref} same={same} diff={diff} max={ {k: max(v) for k, v in mx.items()} }")
PY
    echo "RUN_SUMMARY zero_shot_timing rc=0 probes=$(ls "$R"/pred_"$REF"/*.npz | wc -l) wall_s=$(( $(date +%s) - T0 ))"
    exit 0
fi
python tools/zero_shot_parent.py --parent "$PARENT" --save-dir "$R/pred" --split "$SPLIT" --signed-magnitude "$R"/recipes/*.yaml
rc=$?
echo "ZS_TIMING_WALL $(( $(date +%s) - T0 )) s"
[ "$rc" = 0 ] || { echo "RUN_SUMMARY zero_shot_timing rc=$rc stage=read"; exit "$rc"; }
# PROFILE=K: the first K probes read again under cProfile into a folder of their own; the calls by cumulative time
if [ -n "$PROFILE" ]; then
    mkdir "$R/pred_prof" || exit 1
    python -m cProfile -o "$R/prof.out" tools/zero_shot_parent.py --parent "$PARENT" --save-dir "$R/pred_prof" \
        --split "$SPLIT" --signed-magnitude $(ls "$R"/recipes/*.yaml | head -n "$PROFILE") > "$R/prof_read.log" 2>&1
    echo "ZS_PROFILE rc=$? probes=$PROFILE"
    python -c "
import pstats, sys
s = pstats.Stats(sys.argv[1]); s.sort_stats('cumulative').print_stats(60); s.sort_stats('tottime').print_stats(30)" "$R/prof.out" | sed 's/^/ZS_PROF /'
fi
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
echo "RUN_SUMMARY zero_shot_timing rc=0 probes=$(ls "$R"/pred/*.npz | wc -l) wall_s=$(( $(date +%s) - T0 ))"
