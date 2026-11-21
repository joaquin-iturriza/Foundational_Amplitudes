"""Plan finetune_mechanism step 4: does the W parent's rerun (dsave_parent_finale_zng_uq_w_lr1p89e-3_snap, lxplus,
FA-lxplus-e8d15505) reproduce the original (dsave_parent_finale_zng_uq_w_lr1p89e-3_cc, CC, FA-ccin2p3-47e14eae)?

Per process, for each parent's final model (post-sigma, D276): the best-checkpoint val_loss_no_reg (het. NLL) of the
main stage and of the sigma stage (best_checkpoint), and the MSE of the standardised log|M|^2 prediction over the
sigma stage's saved val predictions (preds_val.npz, the final model). Read on each parent's own site (site_python).
Writes analysis/finetune_mechanism/data/parent_rerun_check.json.

    python3 analysis/finetune_mechanism/parent_rerun_check.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.common.readers import site_python  # noqa: E402

PARENTS = {"cc": ("ccin2p3", "runs/data_saving/dsave_parent_finale_zng_uq_w_lr1p89e-3_cc", "FA-ccin2p3-47e14eae"),
           "snap": ("lxplus", "runs/data_saving/dsave_parent_finale_zng_uq_w_lr1p89e-3_snap", "FA-lxplus-e8d15505")}

REMOTE = r'''
import json, os, sys
import numpy as np
d = sys.argv[1]
def best(f):
    m = json.load(open(f)); i = int(np.argmin(m["val_loss_no_reg"]))
    return {"idx": i, "n_val": len(m["val_loss_no_reg"]), "combined": float(m["val_loss_no_reg"][i]),
            "proc": {n: float(v[i]) for n, v in m["proc_val_losses_no_reg"].items() if len(v) > i}}
out = {"main": best(os.path.join(d, "plots_0", "per_process_metrics.json")),
       "sigma": best(os.path.join(d, "sigma_stage", "plots_0", "per_process_metrics.json"))}
z = np.load(os.path.join(d, "sigma_stage", "preds_val.npz"))
names, pid = list(z["names"]), z["process_id"]
r2 = ((z["pred"][:, 0].astype(np.float64) - z["truth"][:, 0]) ** 2)
out["mse"] = {str(n): float(r2[pid == k].mean()) for k, n in enumerate(names) if (pid == k).any()}
out["n_val"] = {str(n): int((pid == k).sum()) for k, n in enumerate(names)}
print("PRC " + json.dumps(out))
'''


def main():
    res = {}
    for k, (site, d, run) in PARENTS.items():
        res[k] = site_python(REMOTE, [d], "PRC", site=site)
        res[k].update(site=site, run_dir=d, run=run)
    a, b = res["cc"], res["snap"]
    procs = sorted(a["mse"], key=lambda n: a["mse"][n])
    assert set(procs) == set(b["mse"]), "the two parents hold different processes"
    assert a["n_val"] == b["n_val"], "the two parents' val pools differ in size"
    ratio = {n: b["mse"][n] / a["mse"][n] for n in procs}
    dnll = {n: b["sigma"]["proc"][n] - a["sigma"]["proc"][n] for n in procs}
    import numpy as np
    r = np.array(list(ratio.values()))
    res["summary"] = {"procs": procs, "mse_ratio_snap_over_cc": ratio, "sigma_nll_snap_minus_cc": dnll,
                      "ratio_min": float(r.min()), "ratio_max": float(r.max()),
                      "ratio_gmean": float(np.exp(np.log(r).mean())),
                      "combined_nll": {"cc": a["sigma"]["combined"], "snap": b["sigma"]["combined"]}}
    out = os.path.join(os.path.dirname(__file__), "data", "parent_rerun_check.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    s = res["summary"]
    print(f"combined sigma-stage NLL: cc {s['combined_nll']['cc']:.4f} snap {s['combined_nll']['snap']:.4f}")
    print(f"MSE ratio snap/cc over {len(procs)} processes: gmean {s['ratio_gmean']:.3f}, "
          f"range [{s['ratio_min']:.3f}, {s['ratio_max']:.3f}]")
    for n in procs:
        print(f"  {n:18s} cc {a['mse'][n]:.3e} snap {b['mse'][n]:.3e} ratio {ratio[n]:.3f} dNLL {dnll[n]:+.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
