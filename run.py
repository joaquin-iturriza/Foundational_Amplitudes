import hydra
from experiment import AmplitudeExperiment


@hydra.main(config_path="config", config_name="amplitudes", version_base=None)
def main(cfg):
    if cfg.exp_type == "amplitudes":
        exp = AmplitudeExperiment(cfg)
    else:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented")

    exp()


if __name__ == "__main__":
    main()