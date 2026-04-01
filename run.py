import warnings
warnings.filterwarnings("ignore", message=".*NumPy 1.x.*")
warnings.filterwarnings("ignore", message=".*_ARRAY_API.*")

import hydra
import torch
from experiment import AmplitudeExperiment


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


if __name__ == "__main__":
    main()