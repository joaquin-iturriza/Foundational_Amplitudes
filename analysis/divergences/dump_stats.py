#!/usr/bin/env python
"""Dump a run's frozen preprocessing stats to a data_stats.json (CPU, login-node safe).

Used to build the SHARED standardization for a fair sampling A/B: run init_data on the
NATIVE (raw RAMBO) pool once, freeze its mom/amp stats, and point every arm at the result
via data.frozen_stats_path -- so raw and flat-log|M|^2 arms optimize the SAME log-amp loss
scale and differ only in sampling density (CLAUDE.md A/B rule #6). init_data is torch-on-CPU
(COM boost, no xformers), so no GPU is needed.
"""
import argparse
import json
import os
import sys

import numpy as np
from omegaconf import OmegaConf, open_dict

WT = "/lustre/fswork/projects/rech/itg/ulm49ia/Foundational_Amplitudes/worktrees/wt-heterosc"
sys.path.insert(0, WT)
from experiment import AmplitudeExperiment  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="a run's config.yaml (defines the pool + preprocessing)")
    ap.add_argument("--out", required=True, help="output data_stats.json")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    with open_dict(cfg):
        cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False
        cfg.use_mlflow = False; cfg.save_source = False; cfg.warm_start_idx = None
        cfg.ema = False; cfg.count_flops = False
        cfg.run_dir = os.path.join(WT, "runs", "_dump_stats_tmp")
        cfg.data.subsample = None
        cfg.fine_tune.pretrained_path = None       # compute FRESH on this pool, don't inherit
        if "frozen_stats_path" in cfg.data:
            cfg.data.frozen_stats_path = None
    exp = AmplitudeExperiment(cfg)
    exp._init(); exp.init_physics(); exp.init_geometric_algebra(); exp.init_data()

    stats = dict(
        mom_div=float(exp.mom_div),
        mom_mean=float(np.atleast_1d(exp.mom_mean)[0]),
        mom_std=float(np.atleast_1d(exp.mom_std)[0]),
        amp_trafos=list(exp.cfg.data.amp_trafos),
        preprocess_per_dataset=bool(exp.cfg.data.get("preprocess_per_dataset", False)),
        prepd_mean=[float(x) for x in np.atleast_1d(exp.prepd_mean)],
        prepd_std=[float(x) for x in np.atleast_1d(exp.prepd_std)],
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=1)
    print(f"wrote {args.out}")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
