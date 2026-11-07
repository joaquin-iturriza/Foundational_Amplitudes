"""The fine_tune keys that shape only how a run starts from its parent (shrink_perturb, start_weights=live,
load_optimizer_state, moment_fill_steps) are cleared where a finished run is loaded as a parent: the HETEROSC sigma
stage (run.sigma_stage_cfg) and the readers (base_experiment.clear_start_only). Without it the sigma stage of
v4_het_wsx_shrinkperturb (FA-ccin2p3-71852916) shrank its own trained model again, and every probe read once more."""
from omegaconf import OmegaConf, open_dict


def _cfg(**ft):
    base = {"pretrained_path": "/p/models/model_run0_best.pt", "lr_scale": 1.0, "layer_decay": 1.0,
            "freeze_blocks": [], "reset_output_head": False}
    base.update(ft)
    return OmegaConf.create({"run_dir": "/tmp/x", "run_name": "x", "warm_start_idx": None, "ema": True,
                             "data": {"amp_trafos": ["log", "standardization"]}, "model": {"net": {"num_heads": 8}},
                             "fine_tune": base,
                             "training": {"loss": "HETEROSC", "heterosc_sigma_stage_steps": 250,
                                          "scheduler": "CosineAnnealingLR"},
                             "evaluation": {}, "plotting": {}})


def test_sigma_stage_clears_start_only_keys():
    from run import sigma_stage_cfg
    for ft in ({"shrink_perturb": [0.4, 0.1]}, {"start_weights": "live", "load_optimizer_state": True}):
        s = sigma_stage_cfg(_cfg(**ft))
        assert s.fine_tune.shrink_perturb is None
        assert s.fine_tune.start_weights == "scored"
        assert s.fine_tune.load_optimizer_state is False
        assert s.fine_tune.moment_fill_steps == 0
        assert s.fine_tune.pretrained_path == "/tmp/x/models/model_run0_best.pt"


def test_clear_start_only_on_a_struct_config_without_the_keys():
    """A reader's config read from an older run that never had the keys: set, no error under open_dict."""
    from base_experiment import clear_start_only
    cfg = _cfg()
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        clear_start_only(cfg.fine_tune)
    assert cfg.fine_tune.shrink_perturb is None and cfg.fine_tune.start_weights == "scored"

