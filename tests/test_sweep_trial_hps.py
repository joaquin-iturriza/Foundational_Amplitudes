"""tools/sweep_trial_hps.py: a trial's searched HPs as overrides, and the job script's GROW_SWEEP filtering of NORM."""
import os
import subprocess
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import sweep_trial_hps as sth  # noqa: E402

SWEEP = "ws_grow_k533_zeroshot_hpo8"
TRIAL = {"fine_tune": {"lr_scale": 0.31, "layer_decay": 0.9}, "ema": True,
         "training": {"muon_weight_decay": 0.02, "regularization_lambda": 1e-08, "cosanneal_warmup_frac": 0.1,
                      "cosanneal_eta_min": 1e-09, "ema_decay": 0.95, "scheduler": "WSD", "wsd_cooldown_frac": 0.2}}


def test_overrides_every_searched_key(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text(yaml.safe_dump(TRIAL))
    got = sth.overrides(SWEEP, str(f))
    keys = [p["name"] for p in sth.sweep_config(SWEEP)["search_space"]]
    assert [g.split("=")[0] for g in got] == ["++" + k for k in keys]
    assert "++fine_tune.lr_scale=0.31" in got and "++ema=true" in got and "++training.scheduler=WSD" in got


def test_missing_key_or_config_fails(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text(yaml.safe_dump({"fine_tune": {"lr_scale": 1.0}}))
    with pytest.raises(SystemExit):
        sth.overrides(SWEEP, str(f))
    with pytest.raises(SystemExit):
        sth.overrides(SWEEP, str(tmp_path / "none.yaml"))
    with pytest.raises(SystemExit):
        sth.sweep_config("no_such_sweep")


def test_norm_filtering_matches_job_script():
    # the loop of scripts/job_catalog_v3_short.sh's GROW_SWEEP block, run on a NORM holding searched keys with and
    # without the ++ prefix: each searched key ends up exactly once, with the trial's value
    src = open(os.path.join(ROOT, "scripts", "job_catalog_v3_short.sh")).read()
    loop = src[src.index('            for kv in "${GHPA[@]}"'):src.index('            OUT="${OUT}_${GROW_SWEEP}')]
    script = ('NORM=(++fine_tune.lr_scale=1.0 ++fine_tune.layer_decay=1.0 ema=false ++training.muon_weight_decay=0.03 '
              '++data.x=1)\nGHPA=(++fine_tune.lr_scale=0.31 ++fine_tune.layer_decay=0.9 ++ema=true '
              '++training.muon_weight_decay=0.02)\n' + loop + 'printf "%s\\n" "${NORM[@]}"\n')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["++data.x=1", "++fine_tune.lr_scale=0.31", "++fine_tune.layer_decay=0.9", "++ema=true",
                   "++training.muon_weight_decay=0.02"]
