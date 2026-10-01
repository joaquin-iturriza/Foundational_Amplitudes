# CPU dry check (not committed): build a ftp config's data up to init_data and print the off-shellness stats
import os, sys, tempfile, yaml, torch
sys.path.insert(0, os.getcwd())
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
c = yaml.safe_load(open(sys.argv[1]))
ov = [f"{k}={os.path.expandvars(str(v)).replace('${PROJECT_DIR}', os.getcwd())}" for k, v in c["fixed_params"].items()]
with initialize_config_dir(config_dir=os.path.join(os.getcwd(), "config"), version_base=None):
    cfg = compose(config_name="amplitudes", overrides=ov)
with open_dict(cfg):
    cfg.train = False; cfg.evaluate = False; cfg.plot = False; cfg.save = False; cfg.use_mlflow = False
    cfg.count_flops = False; cfg.run_dir = tempfile.mkdtemp(dir=os.environ["SCRATCH"])
torch.set_default_dtype(torch.float32)
from experiment import AmplitudeExperiment
e = AmplitudeExperiment(cfg); e._init(); e.init_physics(); e.init_geometric_algebra(); e.init_data()
print("DRY offshell_stats", e._offshell_stats, "prepd_std", list(e.prepd_std))
