"""The small-D fine-tuning study, first round (D120): one method on one probe's cells (analysis/transfer/data/
lowd_ft_spec.json: rungs 1, 7, 9 at D = 10 to 10^2.5), each run as its grid cell's best fine-tune trial ran (the cell's
sweep config's fixed parameters, its steps, the trial's HPs, cold start, run.py as sweep/run_trial.py builds it) plus the
method's flags:
  default        the grid's fine-tune as it is (full retraining, the cell's layer decay), rerun here on the same
                 hardware as the other methods
  lpft           linear probe, then fine-tune: linear_out fitted on the frozen trunk first (fine_tune.linear_probe)
  l2sp           the fine-tune pulled toward the start: (lambda/2)|theta - theta*|^2 on every weight
                 (fine_tune.ewc.enabled with a uniform Fisher), lambda = 1e-2, not tuned (the DyHPO round tunes it)
  freeze         the lower half of the trunk frozen (fine_tune.freeze_blocks [0,1,2,3] of 8)
One LOWD JSON line per cell: the best-checkpoint val_loss_no_reg (the run's result file), times prepd_std^2, i.e. the
MSE of log|M|^2 as cells.py reports a cell, next to the cell's recorded fine-tune and scratch values.
    python tools/lowd_ft.py --method lpft --probe ee_Za --out $SCRATCH/zz_lowd
"""
import argparse, json, os, subprocess, sys, time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import siteconf  # noqa: E402

METHODS = {
    "default": [],
    "lpft": ["fine_tune.linear_probe=true"],
    "l2sp": ["fine_tune.ewc.enabled=true", "fine_tune.ewc.uniform=true", "fine_tune.ewc.lambda=1e-2"],
    "freeze": ["fine_tune.freeze_blocks=[0,1,2,3]"],
}


def command(cell, method, run_dir, result_path):
    sw = siteconf.resolve(yaml.safe_load(open(os.path.join(ROOT, cell["config"]))))
    T = cell["steps"]
    cmd = [sys.executable, os.path.join(ROOT, "run.py"), "local=none"]
    cmd += [f"{k}={v}" for k, v in sw["fixed_params"].items() if k != "training.iterations"]
    cmd += [f"{k}={v}" for k, v in cell["hp"].items()]
    cmd += [f"training.iterations={T}", f"training.increment_steps={T}", "training.is_dyhpo_run=true",
            f"base_dir={sw['paths']['project_dir']}", f"run_dir={run_dir}", f"exp_name=tp3_lowd_{method}_{cell['probe']}",
            "warm_start_idx=null", f"training.result_path={result_path}"]
    return cmd + METHODS[method]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="cells by tag, e.g. r7_d3 (the GPU smoke test)")
    a = ap.parse_args()
    spec = [c for c in json.load(open(os.path.join(ROOT, "analysis", "transfer", "data", "lowd_ft_spec.json")))
            if c["probe"] == a.probe]
    assert spec, f"no cells for {a.probe}"
    base = os.path.join(a.out, f"tp3_lowd_{a.method}_{a.probe}")
    os.makedirs(base, exist_ok=True)
    for c in spec:
        tag = f"r{c['rung']}_d{c['k']}"
        if a.only and tag not in a.only:
            continue
        run_dir, res = os.path.join(base, tag), os.path.join(base, tag + "_result.json")
        rec = {k: c[k] for k in ("probe", "rung", "k", "D", "steps", "best_hp", "recorded_ft", "scratch")}
        rec["method"] = a.method
        t0 = time.time()
        try:
            p = subprocess.run(command(c, a.method, run_dir, res), cwd=ROOT)
            if p.returncode:
                raise RuntimeError(f"run.py exit {p.returncode}")
            r = json.load(open(res))
            std2 = float(json.load(open(os.path.join(run_dir, "data_stats.json")))["prepd_std"][0]) ** 2
            rec.update(val_loss_no_reg=float(r["val_loss"]), mse_logM2=float(r["val_loss"]) * std2,
                       prepd_std=std2 ** 0.5, best_step=r.get("best_step"), seconds=round(time.time() - t0))
        except Exception as e:                    # one failed cell is listed, never silently dropped
            rec["error"] = repr(e)[:300]
        print("LOWD " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
