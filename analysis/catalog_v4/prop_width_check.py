"""The Higgs width in the propagator read's log P (FA-ccin2p3-71a96ca2 ran with 0.00407 GeV; the sm card's 6.382339e-3 since
97c43af9): per pool with an internal Higgs line among the control's H2 and TR reads, log P at both widths on the read's
events; prints the largest per-event shift, the share of events moved by more than 1e-3 and the least r^2 between the two.
Every read of the design is an r^2, an eta^2 or a decile share, so an r^2 near 1 leaves them unchanged. On CC:
    python analysis/catalog_v4/prop_width_check.py"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, ".")
import analysis.catalog_v4.dipole_residual as dr
from analysis.common.readers import split_pool
out = []
for ax in ("H2", "TR"):
    for f in sorted(glob.glob(os.environ["SCRATCH"] + "/zz_v4_short_v4_dimensionless_muon/" + ax + "/v4_*.npz")):
        name = os.path.basename(f)[:-4]
        sc = "data/diagrams/" + name + ".diagrams.json"
        if not os.path.exists(sc) or not any(abs(e["pdg"]) == 25 and not e["external"] for s_ in json.load(open(sc))["subprocesses"] for d in s_["diagrams"] for e in d["edges"]):
            continue
        z = np.load(f); pr = np.load(str(z["pool"]), mmap_mode="r"); rows = dr.pool_rows(z)
        if rows is None: continue
        mom, pdg, _ = split_pool(np.asarray(pr[rows]), (pr.shape[1] - 1) // 5)
        dr.MW[25] = (125.0, 6.382339e-03); a = dr.log_prop(mom, pdg, sc)
        dr.MW[25] = (125.0, 0.00407); b = dr.log_prop(mom, pdg, sc)
        if a is None: continue
        d = np.abs(a - b)
        out.append((ax, name, float(d.max()), float(np.mean(d > 1e-3)), float(np.corrcoef(a, b)[0, 1] ** 2)))
print(len(out), "pools with an internal Higgs")
print("max |dlogP| over all:", max(o[2] for o in out) if out else None)
print("worst 5:", sorted(out, key=lambda o: -o[2])[:5])
print("min r2(logP_new, logP_old):", min(o[4] for o in out) if out else None)
