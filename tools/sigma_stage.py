"""The sigma-only stage (D70) on an existing HETEROSC run trained before run.py ran it by default: the run's own config,
warm-started from its best checkpoint, trunk and mu frozen, written to <run_dir>/sigma_stage (run.sigma_stage_cfg).
Needs a GPU.
    python tools/sigma_stage.py runs/<sweep>/trial_N [...]
"""
import os, sys

import torch
from omegaconf import OmegaConf, open_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import siteconf  # noqa: E402
from experiment import AmplitudeExperiment  # noqa: E402
from run import sigma_stage_cfg  # noqa: E402

for rd in sys.argv[1:]:
    rd = os.path.abspath(rd)
    if os.path.exists(os.path.join(rd, "sigma_stage", "models")):
        print(f"{rd}: sigma stage already there"); continue
    cfg = OmegaConf.load(os.path.join(rd, "config.yaml"))
    with open_dict(cfg):
        cfg.run_dir = rd
        cfg.base_dir = siteconf.PROJECT_DIR
        if cfg.data.get("processes_file"):
            cfg.data.processes_file = siteconf._expand_str(str(cfg.data.processes_file))
    torch.set_default_dtype({"float64": torch.float64, "float16": torch.float16}.get(str(cfg.training.dtype), torch.float32))
    stage = sigma_stage_cfg(cfg)
    assert stage is not None, f"{rd}: not a joint HETEROSC run"
    with open_dict(stage):
        stage.training.is_dyhpo_run = False
    AmplitudeExperiment(stage)()
