"""The resolved job config of a sweep trial (run_trial.build_command's overrides on config/, composed by Hydra without
running) against an existing run's config.yaml: every key whose value differs, is missing on one side, or is new.
CLAUDE.md: a run built on another run's setup lists every inherited setting that differs from that run's config.yaml.
    python tools/cfg_diff.py --sweep-config sweep/X.yaml --t-steps 60840 --hp training.lr=1e-3 ... --against runs/.../config.yaml
Keys that only name the run's place (base_dir, run_dir, exp_name, training.result_path, run_name) are left out.
"""
import argparse, os, sys

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sweep"))
sys.path.insert(0, ROOT)
from run_trial import build_command  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "tools"))
from restore_run_config import diff  # noqa: E402

PLACE = {"base_dir", "run_dir", "exp_name", "training.result_path", "run_name"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-config", required=True)
    ap.add_argument("--t-steps", type=int, required=True)
    ap.add_argument("--hp", action="append", default=[], help="k=v, as trial_job.sh --hp")
    ap.add_argument("--against", required=True, help="a run's config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.sweep_config))
    cfg["paths"] = {**(cfg.get("paths") or {}), "project_dir": ROOT}
    assert not cfg.get("base_config"), "base_config sweeps: compare against the base run directly"
    hp = dict(kv.split("=", 1) for kv in a.hp)
    cmd = build_command(cfg, hp, "RUN_DIR", 0, "RESULT", a.t_steps, increment_steps=a.t_steps)
    overrides = [c for c in cmd[2:]]
    with initialize_config_dir(config_dir=os.path.join(ROOT, "config"), version_base=None):
        job = OmegaConf.to_container(compose(config_name="amplitudes", overrides=overrides), resolve=False)
    ref = yaml.safe_load(open(a.against))
    d = {k: v for k, v in diff(job, ref).items() if k not in PLACE}
    for k, (x, y) in d.items():
        print(f"{k}: job={x!r} run={y!r}")
    print(f"CFG_DIFF {len(d)} keys differ")


if __name__ == "__main__":
    main()
