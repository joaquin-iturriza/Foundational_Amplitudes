"""Plan finetune_mechanism step 4: does the W parent's rerun (dsave_parent_finale_zng_uq_w_lr1p89e-3_snap, lxplus,
FA-lxplus-e8d15505) reproduce the original (dsave_parent_finale_zng_uq_w_lr1p89e-3_cc, CC, FA-ccin2p3-47e14eae)?

Per process, for each parent's final model (post-sigma, D276): the best-checkpoint val_loss_no_reg (het. NLL) of the
main stage and of the sigma stage (best_checkpoint), and the MSE of the standardised log|M|^2 prediction over the
sigma stage's saved val predictions (preds_val.npz, the final model). Read on each parent's own site (site_python).
Writes analysis/finetune_mechanism/data/parent_rerun_check.json.

    python3 analysis/finetune_mechanism/parent_rerun_check.py [snap|snapfp32|noset]   (the rerun read against cc; default snap,
    the TF32 rerun; snapfp32 = the strict-fp32 rerun, written to parent_rerun_check_snapfp32.json)
    python3 analysis/finetune_mechanism/parent_rerun_check.py glu --ref noset   (the gluon/Z parent against the 28-process
    no-settings parent on the processes they share)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from analysis.common.readers import site_python  # noqa: E402

PARENTS = {"cc": ("ccin2p3", "runs/data_saving/dsave_parent_finale_zng_uq_w_lr1p89e-3_cc", "FA-ccin2p3-47e14eae"),
           "snap": ("lxplus", "runs/data_saving/dsave_parent_finale_zng_uq_w_lr1p89e-3_snap", "FA-lxplus-e8d15505"),
           "snapfp32": ("ccin2p3", "runs/data_saving/dsave_parent_finale_zng_uq_w_lr1p89e-3_snapfp32", "FA-ccin2p3-74805752"),
           # step 5: the parent retrained without input_soft_eps/input_pair_logs (strict fp32, H100)
           "noset": ("ccin2p3", "runs/data_saving/dsave_parent_finale_zng_uq_w_noset_lr1p89e-3", "FA-ccin2p3-9a73698a"),
           # step 2 (H2 dilution): the no-settings parent on the finale's 9 gluon/Z processes only, read --ref noset
           "glu": ("ccin2p3", "runs/data_saving/dsave_parent_finale_zng_uq_w_noset_glu_lr1p89e-3", "FA-ccin2p3-1f2d9ad5")}

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
out["std"] = json.load(open(os.path.join(d, "data_stats.json")))["prepd_std"][0]
print("PRC " + json.dumps(out))
'''


def main():
    res = {}
    key = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "snap"
    ref = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else "cc"
    out = os.path.join(os.path.dirname(__file__), "data", "parent_rerun_check%s.json" % ("" if key == "snap" else "_" + key))
    if "--cached" in sys.argv:   # re-read the saved site reads (verify_claims reruns without ssh)
        res = {k: json.load(open(out))[k] for k in (ref, key)}
    for k in (ref, key):
        if k in res:
            continue
        site, d, run = PARENTS[k]
        res[k] = site_python(REMOTE, [d], "PRC", site=site)
        res[k].update(site=site, run_dir=d, run=run)
    a, b = res[ref], res[key]
    procs = sorted(a["mse"], key=lambda n: a["mse"][n])
    if ref == "cc":
        assert set(procs) == set(b["mse"]), "the two parents hold different processes"
    else:   # a parent on a subset of the reference's processes: compared on the shared ones
        procs = [n for n in procs if n in b["mse"]]
        assert set(procs) == set(b["mse"]), "the parent holds a process the reference lacks"
    assert all(a["n_val"][n] == b["n_val"][n] for n in procs), "the two parents' val pools differ in size"
    # MSE of log|M|^2: each parent standardises with its own target std (different process sets give different stds)
    k2 = (b["std"] / a["std"]) ** 2 if ref != "cc" else 1.0
    ratio = {n: b["mse"][n] * k2 / a["mse"][n] for n in procs}
    dnll = {n: b["sigma"]["proc"][n] - a["sigma"]["proc"][n] for n in procs}
    import numpy as np
    r = np.array(list(ratio.values()))
    res["summary"] = {"procs": procs, "mse_ratio_snap_over_cc": ratio, "sigma_nll_snap_minus_cc": dnll,
                      "ratio_min": float(r.min()), "ratio_max": float(r.max()),
                      "ratio_gmean": float(np.exp(np.log(r).mean())),
                      "combined_nll": {ref: a["sigma"]["combined"], key: b["sigma"]["combined"]}, "rerun": key, "ref": ref}
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    s = res["summary"]
    print(f"combined sigma-stage NLL: {ref} {s['combined_nll'][ref]:.4f} {key} {s['combined_nll'][key]:.4f}")
    print(f"MSE ratio {key}/{ref} over {len(procs)} processes: gmean {s['ratio_gmean']:.3f}, "
          f"range [{s['ratio_min']:.3f}, {s['ratio_max']:.3f}]")
    for n in procs:
        print(f"  {n:18s} {ref} {a['mse'][n]:.3e} {key} {b['mse'][n]:.3e} ratio {ratio[n]:.3f} dNLL {dnll[n]:+.3f}")
    print("wrote", out)
    print(json.dumps({"ratio_gmean": s["ratio_gmean"], "n_procs": len(procs),
                      "n_within_0p43_2p5": int(((r >= 0.43) & (r <= 2.5)).sum()),
                      "combined_nll": s["combined_nll"], "ratio": ratio}))


if __name__ == "__main__":
    main()
