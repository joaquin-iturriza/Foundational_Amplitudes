#!/usr/bin/env python3
"""A sweep trial's searched HPs as Hydra overrides, for re-running the trial's setting in a job (plan warm_start step
11: scripts/job_catalog_v3_short.sh GROW_SWEEP/GROW_HP). The searched keys are the search_space of the sweep config
under sweep/ whose sweep_name is <sweep>; their values are read from the trial's own saved config (the resolved
config.yaml run.py wrote), so the re-run gets exactly what the trial ran with.
    python tools/sweep_trial_hps.py <sweep_name> <trial dir>/config.yaml   -> one line: ++k=v ++k=v ...
Exits 1 (message on stdout) when the config, the trial's config or a key is missing."""
import glob
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sweep_config(name):
    hits = []
    for f in sorted(glob.glob(os.path.join(ROOT, "sweep", "sweep_config*.yaml"))):
        try:
            c = yaml.safe_load(open(f))
        except yaml.YAMLError:
            continue
        if isinstance(c, dict) and c.get("sweep_name") == name:
            hits.append((f, c))
    if len(hits) != 1:
        raise SystemExit(f"sweep_trial_hps: {len(hits)} sweep configs named {name}")
    return hits[0][1]


def fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return repr(v) if isinstance(v, float) else str(v)


def overrides(name, trial_cfg):
    keys = [p["name"] for p in sweep_config(name)["search_space"]]
    if not os.path.exists(trial_cfg):
        raise SystemExit(f"sweep_trial_hps: no {trial_cfg}")
    cfg = yaml.safe_load(open(trial_cfg))
    out = []
    for k in keys:
        x = cfg
        for part in k.split("."):
            if not isinstance(x, dict) or part not in x:
                raise SystemExit(f"sweep_trial_hps: {k} not in {trial_cfg}")
            x = x[part]
        out.append(f"++{k}={fmt(x)}")
    return out


if __name__ == "__main__":
    try:
        print(" ".join(overrides(sys.argv[1], sys.argv[2])))
    except SystemExit as e:          # the reason on stdout, where the job script's $(...) captures it
        print(e.code)
        sys.exit(1)
