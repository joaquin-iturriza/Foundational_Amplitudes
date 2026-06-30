#!/usr/bin/env python
"""Coupling capability honest-test recipe.

K copies of ee_uuggg (LO, α_s^3 — maximal coupling leverage) at WIDE *flat* α_s
values. For LO, physics.alpha_s patches the param_card α_s with NO running, and a
pure coupling scan leaves the phase space bit-identical to the base (same seed),
so all K datasets have IDENTICAL kinematics and differ ONLY by the α_s^3 amplitude
factor. That is the decoupled-α test: at fixed kinematics the amplitude varies
purely with α.

Run it POOLED (data.preprocess_per_dataset=false) so the α-offset is NOT removed
by per-dataset standardization — then:
  - off arm  (coupling_scalars=false): identical kinematics + identical amp_orders
    [0,3] across all K, so it CANNOT tell which α a given event used -> must
    regress to the mean -> irreducible error = spread of 3·ln(α).
  - feature arm (coupling_scalars=true): gets log(α) per dataset -> can resolve it.
If the feature wins here, the architecture CAN use coupling; the production tie is
then the per-dataset-standardization × per-dataset-granularity interaction, not an
architecture limit. If it does NOT win even here, the net can't use the feature.
"""
import numpy as np, yaml, os

K = 8
ALPHAS = np.round(np.geomspace(0.05, 0.40, K), 4)   # wide, ~decade; 3·ln spread ≈ 6.2
SQRTS = [91, 1000]
N_TRAIN, N_VAL, N_TEST = 6000, 1500, 1500

procs = [{
    "name": f"ee_uuggg_pa{i:02d}",
    "base": "ee_uuggg",
    "sqrts": SQRTS,
    "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
    "physics": {"alpha_s": float(a)},
} for i, a in enumerate(ALPHAS)]

out = os.path.join(os.path.dirname(__file__), "coupling_capability_test.yaml")
with open(out, "w") as f:
    yaml.safe_dump({"processes": procs}, f, default_flow_style=True, sort_keys=False)
print(f"wrote {out}: {K} ee_uuggg datasets, flat alpha_s in {ALPHAS[0]}..{ALPHAS[-1]}")
print("3*ln(alpha) offsets:", np.round(3*np.log(ALPHAS), 2))
