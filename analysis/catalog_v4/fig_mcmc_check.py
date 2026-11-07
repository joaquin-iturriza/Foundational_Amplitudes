"""Step 69's figure (plan catalog_coverage): per test class, the flat part's fill per in-range 0.25-dex log10|M|^2 bin
(min(kept, quota) / quota, mean of seeds 42 and 43) for the Markov-chain flat part at 12 labels per flat event, the
quota sampler at 12 and at 50; each legend label carries the policy's cost (train-pool replay seconds over the old
generation-log seconds, mean of the seeds). Reads data/mcmc_check.json (multichannel_check.py step69).
With the argument grid (plan step 72): data/grid_check.json (multichannel_check.py step72), the channel-grid quota
sampler at 12 and 8 labels per flat event against the quota sampler at 12 and 50, figs/grid_check_*."""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import plot_style as ps

GRID = len(sys.argv) > 1 and sys.argv[1] == "grid"
D = json.load(open(os.path.join(ROOT, "analysis", "catalog_v4", "data", "grid_check.json" if GRID else "mcmc_check.json")))
LAB = {"v4_uux_epemg": r"$u\bar u\to e^+e^-g$", "v4_uux_epemvevex": r"$u\bar u\to e^+e^-\nu_e\bar\nu_e$",
       "v4_uu_bbxuu": r"$uu\to b\bar b uu$", "v4_us_ddxsu": r"$us\to d\bar d su$",
       "v4_udx_csxvevex": r"$u\bar d\to c\bar s\nu_e\bar\nu_e$"}
POL = (("mcmc12", "chains, 12", ps.C.vermillion), ("quota12", "quota, 12", ps.C.blue), ("quota50", "quota, 50", ps.C.green))
if GRID:
    POL = (("grid12", "grids, 12", ps.C.vermillion), ("grid8", "grids, 8", ps.C.orange), ("quota12", "quota, 12", ps.C.blue),
           ("quota50", "quota, 50", ps.C.green))
figs = ps.panels(len(D["classes"]))
for (fig, ax), (c, o) in zip(figs, D["classes"].items()):
    for k, name, col in POL:
        rows = [o["seeds"][s][k] for s in o["seeds"]]
        x = np.asarray(rows[0]["lo_in_range"]) + 0.125
        y = np.mean([r["fill_in_range"] for r in rows], 0)
        cost = np.mean([r["cost_vs_old_log"] for r in rows])
        ax.plot(x, y, color=col, label=rf"{name} (${cost:.1f}\times$)")
    ax.set_xlabel(r"$\log_{10}|\mathcal{M}|^2$"); ax.set_ylabel("fill")
    ps.legend(ax, "lower left")
    ps.process_label(ax, LAB.get(c, c))
ps.save_panels(figs, os.path.join(ROOT, "analysis", "catalog_v4", "figs", "grid_check" if GRID else "mcmc_check"))
