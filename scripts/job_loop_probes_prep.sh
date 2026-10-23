#!/bin/bash
#SBATCH --job-name=loop_probes_prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=loop_probes_prep_%j.out
#SBATCH --error=loop_probes_prep_%j.out
# CPU only. The order axis of the dimensionless round (D93): the pools of recipes/catalog_loop_probes.yaml (the positive
# one-loop pools with their trees; existing pools are cache hits, the missing 1000-event tree train pools are built),
# then each pool's per-event alpha_s (tools/write_own_alphas.py) for the runs with data.target_physics_orders.
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PROJECT_DIR"
SPEC="${LOOP_SPEC:-recipes/catalog_loop_probes.yaml}"   # LOOP_SPEC=recipes/catalog_loop_signed.yaml: the signed pools
# prebuild writes a pool whose recipe_id differs (count, seed, cuts) over the existing file of the same name: refuse
# unless every existing pool of the spec is a cache hit, so only the missing pools get built
python - "$SPEC" <<'PY' || exit 1
import os, sys, yaml
import datagen, mg5_pipeline_final as mg
from prebuild_recipes import COUNT_KEY
bad, missing = [], []
for p in yaml.safe_load(open(sys.argv[1]))["processes"]:
    for role in ("val", "test", "train"):
        r = mg.variable_energy_recipe(p["name"], float(p["sqrts"][0]), float(p["sqrts"][1]), int(p[COUNT_KEY[role]]), role=role, seed=42)
        out = mg.recipe_output_path(r, datagen.dest_for_role(role))
        if not os.path.exists(out):
            missing.append((p["name"], role))
        elif not datagen._is_cached(out, mg.recipe_id(r)):
            bad.append((p["name"], role, out))
print(len(missing), "pools to build:", missing)
# prebuild_recipes.py keys a pool to build by (name, role): two missing windows of one name would share its chunk plan
clash = [k for k in set(missing) if missing.count(k) > 1]
if clash:
    print("REFUSED: two missing windows of one process in one role (prebuild keys by name):", clash); sys.exit(1)
if bad:
    print("REFUSED: existing pools would be rebuilt over:", bad); sys.exit(1)
PY
python prebuild_recipes.py "$SPEC" --workers "${SLURM_CPUS_PER_TASK:-16}" --auto-workers || exit 1
python tools/write_own_alphas.py "$SPEC"
