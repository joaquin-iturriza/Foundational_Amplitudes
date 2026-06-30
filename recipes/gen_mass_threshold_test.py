#!/usr/bin/env python
"""Mass honest-test recipe: ee_ttbar with a WIDE top-mass scan, sitting NEAR
THRESHOLD where m_t actually shapes the per-event amplitude (phase space opens as
√s crosses 2·m_t). This is the regime the original ±15% high-√s scan missed (it
gave ~0.13σ). Here m_t moves the threshold a lot, so the mass produces a per-event
SHAPE that survives per-dataset standardization — a fair test of whether feeding
the (per-particle, derived) on-shell mass via data.mass_from_momenta helps vs
leaving it implicit in the momenta.

K datasets, top mass spread widely; scan_sqrts (in the prebuild) raises each
dataset's √s_min to just above its 2·m_t threshold, and √s_max is modest so events
cluster where the mass matters."""
import numpy as np, yaml, os

MT0 = 173.0
K = 8
MT = np.round(np.linspace(140.0, 205.0, K), 3)   # wide: thresholds 280..410 GeV
SQRTS_MAX = 700.0                                  # modest, so near-threshold events dominate
N_TRAIN, N_VAL, N_TEST = 6000, 1500, 1500

procs = []
for i, mt in enumerate(MT):
    procs.append({
        "name": f"ee_ttbar_mt{i:02d}",
        "base": "ee_ttbar",
        "sqrts": [round(2*float(mt)*1.02, 1), SQRTS_MAX],   # just above 2 m_t
        "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
        "physics": {"masses": {6: float(mt)}},
    })

spec = {"processes": procs}
out = os.path.join(os.path.dirname(__file__), "mass_threshold_test.yaml")
with open(out, "w") as f:
    yaml.safe_dump(spec, f, default_flow_style=True, sort_keys=False)
print(f"wrote {out}: {len(procs)} ee_ttbar datasets, m_t in {MT[0]}..{MT[-1]} GeV")
print("thresholds 2*m_t =", [round(2*m,0) for m in MT])
