#!/usr/bin/env python
"""Reset the dynamic DyHPO state of the scalar A/B arm to fresh, keeping the exact
candidate set (so the re-sweep with the divergence-abort + failure-penalty fixes is
directly comparable to off/diagram). Clears observations, diverged_configs, in_flight,
histories, budget — everything the sampler accumulates — but leaves candidates_raw/
candidates_array/hp_space/seed untouched."""
import pickle, sys, os, shutil

P = "sweeps/scan_ab/scan_ab_scalar/dyhpo_state.pkl"
s = pickle.load(open(P, "rb"))

DYNAMIC = {
    "observations": {}, "diverged_configs": set(), "in_flight": set(),
    "val_loss_history": {}, "proc_val_loss_history": {}, "eval_order": [],
    "budget_spent": 0, "no_improvement_patience": 0,
    "best_value_observed": float("-inf"),   # algorithm fresh-init is np.NINF, NOT None
    "extension_count": {},
}
before = {k: (len(v) if hasattr(v, "__len__") else v)
          for k, v in s.items() if k in DYNAMIC}
for k, v in DYNAMIC.items():
    if k in s:
        s[k] = v if not isinstance(v, dict) else dict(v)
# keep init_conf_indices / initial_random_index as-is (they index the fixed candidates)

shutil.copy(P, P + ".prefix_bak")
pickle.dump(s, open(P, "wb"))
# also wipe stale per-trial result JSONs + checkpoint index so nothing is reused
rd = "sweeps/scan_ab/scan_ab_scalar/results"
for f in os.listdir(rd):
    if f.endswith(".json"):
        os.remove(os.path.join(rd, f))
ckpt = "sweeps/scan_ab/scan_ab_scalar/checkpoint_index.json"
if os.path.exists(ckpt):
    open(ckpt, "w").write("{}")
print("reset scalar DyHPO state; cleared:", before)
print("kept candidates:", len(s.get("candidates_raw", [])))
