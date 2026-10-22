import warnings
warnings.filterwarnings("ignore", message=".*NumPy 1.x.*")
warnings.filterwarnings("ignore", message=".*_ARRAY_API.*")

import os

import hydra
import torch
from omegaconf import OmegaConf, open_dict
from experiment import AmplitudeExperiment


def sigma_stage_cfg(cfg):
    """The config of the sigma-only stage that ends a HETEROSC run (training.heterosc_sigma_stage_steps, D70): the
    run's own config, warm-started from its best checkpoint, with the trunk and mu frozen (heterosc_sigma_only) and the
    D69 sweep's best settings. None when the run is not a joint HETEROSC fit or the stage is off."""
    t = cfg.training
    steps = int(t.get("heterosc_sigma_stage_steps", 250))     # 250 also for a config written before the key (D70)
    if (str(t.get("loss")) != "HETEROSC" or steps <= 0 or t.get("heterosc_sigma_only", False)
            or t.get("heterosc_mu_only", False) or not cfg.get("run_dir")):
        return None
    s = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(s):
        s.run_dir = os.path.join(cfg.run_dir, "sigma_stage")
        s.run_name = None
        s.warm_start_idx = None
        s.ema = False
        s.fine_tune.pretrained_path = os.path.join(cfg.run_dir, "models", "model_run0_best.pt")
        s.fine_tune.lr_scale = 1.0
        s.fine_tune.layer_decay = 1.0
        s.fine_tune.freeze_blocks = []
        s.fine_tune.reset_output_head = False
        s.training.heterosc_sigma_only = True
        s.training.heterosc_select = "nll"          # mu is frozen here, so only the NLL can tell the steps apart
        s.training.lr = float(t.get("heterosc_sigma_stage_lr", 0.01) or 0.01)
        s.training.iterations = steps
        s.training.increment_steps = steps
        s.training.cosanneal_warmup_frac = 0.0
        s.training.regularization_lambda = 0.0
        s.training.result_path = None
        # 25 validations by default (every 10 steps of 250, as in the D69 sweep); fewer for short runs
        s.training.validate_frac = 1.0 / int(t.get("heterosc_sigma_stage_validations", 25) or 25)
        # the stage's evaluation writes the calibrated model's per-event predictions and sigma (preds_<split>.npz in
        # sigma_stage/), so a het trial needs no separate evaluation job for its Delta plots and pulls
        s.evaluation.save_predictions = True
    return s


@hydra.main(config_path="config", config_name="amplitudes", version_base=None)
def main(cfg):
    if cfg.exp_type == "amplitudes":
        exp = AmplitudeExperiment(cfg)
    else:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented")

    match cfg.training.dtype:
        case 'float16':
            torch.set_default_dtype(torch.float16)
        case 'float64':
            torch.set_default_dtype(torch.float64)
        case 'float32':
            torch.set_default_dtype(torch.float32)
        case _:
            raise ValueError(f"dtype {cfg.dtype} not implemented")

    exp()
    stage = sigma_stage_cfg(exp.cfg)
    if stage is not None:
        # the stage trains on the data just used: reuse it in this process instead of reading every pool again
        # (on the catalog the reload was most of the stage's 44 min); the trained model is freed first
        exp.model = exp.optimizer = exp.scheduler = exp.ema = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        nxt = AmplitudeExperiment(stage)
        nxt._reuse_data_from = exp
        nxt()


if __name__ == "__main__":
    main()