#!/bin/bash
#SBATCH --job-name=summary_probe
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# CPU infrastructure check of the setup checks (no training): prints the --env variables it received and the
# RUN_SUMMARY line of a configuration that reproduces the 2026-10-15 bug, a 16k run carrying a DyHPO trial's
# increment_steps=8000, through the real BaseExperiment._init_scheduler and _print_run_summary. `site supervise`
# must flag its T_max (8000, expected 16000).
_CCORCH_ROOT="${CCORCH_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}}"
source "$_CCORCH_ROOT/sites/activate.sh"
cd "$PROJECT_DIR"
echo "site=$CCORCH_SITE host=$(hostname) PARENTS=[${PARENTS:-}] PROBE_NOTE=[${PROBE_NOTE:-}]"
python - <<'EOF'
import types, torch
from omegaconf import OmegaConf
import base_experiment as be
cfg = OmegaConf.merge(OmegaConf.load("config/default.yaml"), {"model": OmegaConf.load("config/model/lloca.yaml")})
cfg.training.iterations = 16000
cfg.training.increment_steps = 8000          # the leftover of a DyHPO trial config
cfg.training.scheduler = "CosineAnnealingLR"
cfg.training.cosanneal_warmup_frac = 0.1
cfg.run_dir = "zz_probe_summary"
o = types.SimpleNamespace(cfg=cfg, warm_start=False, _n_train_events=0,
                          optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3))
be.BaseExperiment._init_scheduler(o)
be.BaseExperiment._print_run_summary(o)
EOF
