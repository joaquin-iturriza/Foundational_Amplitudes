"""Why the het sign head (step 56, arm v4_het_control_signhead) transfers worse than the MSE control's (plan
catalog_coverage step 61, D287/D296): from the runs' job logs (site logs), per run the main block's pre-clip gradient
norm (median, p90, max, share of steps the clip at 5 bounds), the per-validation sign accuracy on the 63 signed training
pools (median, min), the training time per step and the job's wall time; then the het/MSE gradient-norm ratio (by which
the het loss's magnitude gradient outweighs the sign cross-entropy, whose per-event gradient is the same under both
losses at sign_loss_weight 1), and the wrong-sign ratio arm/MSE in distribution (the signed training pools' validation)
against held out (H5, HD5 medians of data/het_signhead.json).

    python3 analysis/catalog_v4/het_sign_diag.py pull    # site logs -> data/reads/<run>_signdiag.json
    python3 analysis/catalog_v4/het_sign_diag.py read    # -> data/het_sign_diag.json
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime

import numpy as np

HERE = os.environ.get("HSD_HERE") or os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
READS = os.path.join(DATA, "reads")
sys.path.insert(0, HERE)
from het_oneloop import SIGNHEAD  # noqa: E402

RUNS = {"het_sign": SIGNHEAD["arm"], "het_ref": SIGNHEAD["ref"], "mse_sign": SIGNHEAD["mse_train"]}
GN = re.compile(r"Grad-norm \(pre-clip\): median=([0-9.e+-]+) p90=([0-9.e+-]+) max=([0-9.e+-]+) \| clip=\S+ bound "
                r"([0-9.]+)% of steps")
ACC = re.compile(r"sign accuracy on the (\d+) signed pools: median ([0-9.]+), min ([0-9.]+)")
FIN = re.compile(r"Finished training: (\d+) iterations .* in ([0-9.]+)min \(avg ([0-9.]+)s/iter\)")
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ")


def pull():
    for kind, runs in RUNS.items():
        for run in runs:
            txt = subprocess.run(["site", "logs", run, "-n", "3000000"], capture_output=True, text=True,
                                 timeout=1800).stdout
            lines = txt.splitlines()
            gn = [m.groups() for m in (GN.search(l) for l in lines) if m]
            fin = [m.groups() for m in (FIN.search(l) for l in lines) if m]
            ts = [m.group(1) for m in (TS.match(l) for l in lines) if m]
            gpu = re.search(r'"gpu": "([^"]+)"', txt)
            rec = {"run": run, "kind": kind, "source": "site logs",
                   # the first Grad-norm line is the main block's, a second one the sigma stage's
                   "grad_norm_main": [float(x) for x in gn[0]] if gn else None,
                   "grad_norm_sigma_stage": [float(x) for x in gn[1]] if len(gn) > 1 else None,
                   "val_sign_acc": [[int(m.group(1)), float(m.group(2)), float(m.group(3))]
                                    for m in (ACC.search(l) for l in lines) if m and int(m.group(1)) > 1],
                   "train_main": [int(fin[0][0]), float(fin[0][1]), float(fin[0][2])] if fin else None,
                   "wall_h": round((datetime.fromisoformat(ts[-1]) - datetime.fromisoformat(ts[0])).total_seconds()
                                   / 3600, 3) if ts else None,
                   "gpu": gpu.group(1) if gpu else None}
            json.dump(rec, open(os.path.join(READS, f"{run}_signdiag.json"), "w"), indent=0)
            print(run, kind, rec["grad_norm_main"], rec["train_main"], rec["wall_h"], rec["gpu"], flush=True)


def read():
    recs = {k: [json.load(open(os.path.join(READS, f"{r}_signdiag.json"))) for r in runs] for k, runs in RUNS.items()}
    out = {"design": "plan catalog_coverage step 61 diagnosis (D287, D296); runs of step 56 (het_oneloop.SIGNHEAD)",
           "per_run": {k: [{x: r[x] for x in ("run", "grad_norm_main", "train_main", "wall_h", "gpu")} |
                           {"val_sign_acc_last": r["val_sign_acc"][-1] if r["val_sign_acc"] else None}
                           for r in v] for k, v in recs.items()}}
    med = {k: [r["grad_norm_main"][0] for r in v] for k, v in recs.items()}
    out["grad_norm_median"] = med
    out["clip_bound_pct"] = {k: [r["grad_norm_main"][3] for r in v] for k, v in recs.items()}
    ratio = [a / b for a, b in zip(med["het_sign"], med["mse_sign"])]     # per seed, lists in seed order 42-45
    out["het_over_mse_grad_norm"] = {"per_seed": [round(x, 1) for x in ratio],
                                     "geomean": round(float(np.exp(np.mean(np.log(ratio)))), 1)}
    last = {k: [r["val_sign_acc"][-1] for r in v if r["val_sign_acc"]] for k, v in recs.items()}
    ws_in = {k: float(np.mean([1 - x[1] for x in v])) for k, v in last.items() if v}
    ws_min = {k: float(np.mean([1 - x[2] for x in v])) for k, v in last.items() if v}
    sh = json.load(open(os.path.join(DATA, "het_signhead.json")))["sign"]
    held = {ax: (sh[ax]["wrong_sign"]["median_arm_pct"] / 100, sh[ax]["wrong_sign"]["median_mse_ctl_pct"] / 100)
            for ax in ("H5", "HD5")}
    out["wrong_sign"] = {
        "train_pools_val_median_wrong": {k: round(v, 4) for k, v in ws_in.items()},
        "train_pools_val_worst_pool_wrong": {k: round(v, 4) for k, v in ws_min.items()},
        "ratio_arm_over_mse": {"train_median": round(ws_in["het_sign"] / ws_in["mse_sign"], 2),
                               "train_worst": round(ws_min["het_sign"] / ws_min["mse_sign"], 2),
                               **{f"{ax}_median": round(a / c, 2) for ax, (a, c) in held.items()}},
        "note": "train: 1 - the last validation's median (worst-pool) sign accuracy over the 63 signed training pools "
                "(200 events each), seed mean; H5/HD5: the median over pools of the seed-mean wrong-sign share "
                "(het_signhead.json)"}
    out["step_time_s"] = {k: round(float(np.mean([r["train_main"][2] for r in v if r["train_main"]])), 4)
                          for k, v in recs.items()}
    out["wall_h"] = {k: [r["wall_h"] for r in v] for k, v in recs.items()}
    out["gpu"] = {k: sorted({r["gpu"] for r in v}) for k, v in recs.items()}
    json.dump(out, open(os.path.join(DATA, "het_sign_diag.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_run"}, indent=1))


if __name__ == "__main__":
    {"pull": pull, "read": read}[sys.argv[1]]()
